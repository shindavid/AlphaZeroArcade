#pragma once

#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/YieldManager.hpp"
#include "core/concepts/Game.hpp"
#include "mcts/ActionSelector.hpp"
#include "mcts/Constants.hpp"
#include "mcts/ManagerParams.hpp"
#include "mcts/NNEvaluationRequest.hpp"
#include "mcts/NNEvaluationService.hpp"
#include "mcts/NNEvaluationServiceBase.hpp"
#include "mcts/Node.hpp"
#include "mcts/SearchParams.hpp"
#include "mcts/TypeDefs.hpp"
#include "util/Math.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <array>
#include <queue>
#include <vector>

namespace mcts {

/*
 * The Manager class is the main entry point for doing MCTS searches.
 *
 * It maintains the search-tree and manages the threads and services that perform the search.
 */
template <core::concepts::Game Game>
class Manager {
 public:
  using ManagerParams = mcts::ManagerParams<Game>;
  using NNEvaluationRequest = mcts::NNEvaluationRequest<Game>;
  using NNEvaluationService = mcts::NNEvaluationService<Game>;
  using NNEvaluationServiceBase = mcts::NNEvaluationServiceBase<Game>;
  using Node = mcts::Node<Game>;
  using LookupTable = Node::LookupTable;
  using LocalPolicyArray = Node::LocalPolicyArray;
  using node_pool_index_t = Node::node_pool_index_t;
  using Edge = Node::Edge;
  using expansion_state_t = Node::expansion_state_t;
  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;
  using ActionValueTensor = Game::Types::ActionValueTensor;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;

  using ActionSelector = mcts::ActionSelector<Game>;
  using ChanceDistribution = Game::Types::ChanceDistribution;
  using ActionRequest = Game::Types::ActionRequest;
  using GameResults = Game::GameResults;
  using Rules = Game::Rules;
  using Symmetries = Game::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;
  using IO = Game::IO;
  using Constants = Game::Constants;
  using State = Game::State;
  using StateHistory = Game::StateHistory;
  using SearchResults = Game::Types::SearchResults;
  using InputTensorizor = Game::InputTensorizor;
  using MCTSKey = InputTensorizor::MCTSKey;
  using StateHistoryArray = std::array<StateHistory, SymmetryGroup::kOrder>;

  using ValueTensor = Game::Types::ValueTensor;
  using ValueArray = Game::Types::ValueArray;

  using post_visit_func_t = std::function<void()>;

  struct Visitation {
    Node* node;
    Edge* edge;  // emanates from node, possibly nullptr
  };
  using search_path_t = std::vector<Visitation>;

  struct SearchRequest {
    SearchRequest(const core::YieldNotificationUnit& u) : notification_unit(u) {}
    SearchRequest() = default;

    core::YieldManager* yield_manager() const { return notification_unit.yield_manager; }
    core::context_id_t context_id() const { return notification_unit.context_id; }
    core::game_slot_index_t game_slot_index() const { return notification_unit.game_slot_index; }

    core::YieldNotificationUnit notification_unit;
  };

  struct SearchResponse {
    static SearchResponse make_drop() { return SearchResponse(nullptr, core::kDrop); }
    static SearchResponse make_yield(int e = 0) { return SearchResponse(nullptr, core::kYield, e); }

    SearchResponse(const SearchResults* r, core::yield_instruction_t y = core::kContinue, int e = 0)
        : results(r), yield_instruction(y), extra_enqueue_count(e) {}

    const SearchResults* results;
    core::yield_instruction_t yield_instruction;
    int extra_enqueue_count;
  };

  struct SearchContext {
    int log_prefix_n() const { return kThreadWhitespaceLength * id; }

    core::context_id_t id;

    search_path_t search_path;

    NNEvaluationRequest eval_request;
    StateHistory canonical_history;
    StateHistoryArray root_history_array;
    StateHistory raw_history;
    core::seat_index_t active_seat;
    group::element_t canonical_sym;

    bool mid_expansion = false;
    bool mid_visit = false;
    bool mid_search_iteration = false;
    bool mid_node_initialization = false;
    bool in_visit_loop = false;

    // node-initialization yield info
    StateHistory* initialization_history;
    node_pool_index_t initialization_index = -1;
    node_pool_index_t inserted_node_index = -1;
    bool expanded_new_node = false;

    // visit yield info
    Node* visit_node;
    Edge* visit_edge;
    StateHistory* history;
    group::element_t inv_canonical_sym;
    bool applied_action = false;

    // For kYield responses
    core::slot_context_vec_t pending_notifications;
    int pending_notifications_mutex_id = 0;

    // For convenience
    const SearchRequest* search_request = nullptr;
  };

  enum execution_state_t : int8_t { kIdle, kInitializingRoot, kInVisitLoop };

  // StateMachine is used to track the state of the Manager's execution.
  //
  // In the GameServer GameThread/GameSlot paradigm, there are a set of GameThread's that
  // continuously cycle through a set of GameSlot's. This translates to calls to Manager::search().
  // If the Manager is configured for multithreading, there can be many such concurrent calls.
  //
  // The Manager must carefully coordinate these calls, so that in general (N - 1) of those calls
  // yield, while the N'th call returns results. It should do this in such a way so that typically,
  // none of those N calls end up blocked waiting for an nn-eval (assuming the GameServer is
  // appropriately configured with enough GameSlot's). This coordination is a sensitive task.
  //
  // The Manager maintains a list of C SearchContext's, where C is its configured concurrency level
  // (in training, C=1, while in competition, C>1). It also maintains an execution_state_t enum
  // that tracks the execution state.
  //
  // Each incoming search() comes with an assigned context id, which is used to select a
  // SearchContext to work with. Here is a summary of the state machine:
  //
  // ** kIdle
  //
  // This is the entry point. This state should only get hit with context 0. After transitioning to
  // kInitializingRoot and releasing the mutex, it does the work of intializing the root. This could
  // require a nn-eval (when AV-targets are needed at the root), which can cause a yield. If it does
  // not yield, it completes root initialization and then, after re-acquiring the mutex, transitions
  // to kInVisitLoop.
  //
  // ** kInitializingRoot
  //
  // This state should also only get hit with context 0. This state indicates that the context
  // previously started the root initialization work and decided to yield. The fact that we enter
  // again here indicates that the nn eval is done. We complete the root initialization and then
  // transition to kInVisitLoop. All of this is done without releasing the mutex.
  //
  // ** kInVisitLoop
  //
  // This is the main work of the search. Context 0 is the first to enter this state, and when doing
  // so, it requests the GameServer to enqueue (C-1) extra slots to its queue, thus activating the
  // multithreaded search.
  //
  // On the first call, it starts an MCTS iteration. If it reaches a leaf node where it requires an
  // nn-eval, it yields. Otherwise, it completes the MCTS iteration and repeats. Eventually, it
  // should either get to a yield point or hit the termination condition.
  //
  // If it hits a yield point, we record this in the SearchContext, so that on the subsequent visit,
  // it can resume from the yield point. This should be the case for all visits in the kInVisitLoop
  // state except for the first.
  //
  // If the termination condition is met, either because the current search iteration triggers it
  // or because it was already met in a previous search iteration, we decrement in_visit_loop_count.
  // If it reaches 0, we transition to kIdle and return results. Otherwise, we issue a kDrop to the
  // caller, which will cause the GameServer to drop the context.
  struct StateMachine {
    mutable mit::mutex mutex;
    int16_t in_visit_loop_count = 0;
    execution_state_t state = kIdle;
  };

  struct RootInfo {
    StateHistoryArray history_array;

    group::element_t canonical_sym = -1;
    node_pool_index_t node_index = -1;
    core::seat_index_t active_seat = -1;
    bool add_noise = false;
  };

  /*
   * Construct a Manager object.
   *
   * Can optionally pass in an NNEvaluationService object. This is useful to pass in a mock service
   * for testing.
   *
   * Can optionally pass mutex_pool's to be used for Node's and SearchContext's. If not provided,
   * the Manager will create a separate single-element mutex-pool for each.
   */
  Manager(const ManagerParams&, core::GameServerBase* server = nullptr,
          NNEvaluationServiceBase* service = nullptr);

  Manager(mutex_vec_sptr_t& node_mutex_pool, mutex_vec_sptr_t& context_mutex_pool,
          const ManagerParams& params, core::GameServerBase* server = nullptr,
          NNEvaluationServiceBase* service = nullptr);

  ~Manager();

  const ManagerParams& params() const { return params_; }
  int num_search_threads() const { return params().num_search_threads; }
  bool multithreaded() const { return num_search_threads() > 1; }

  void start();
  void clear();
  void receive_state_change(core::seat_index_t, const State&, core::action_t);
  void update(core::action_t);

  void set_search_params(const SearchParams& search_params);
  SearchResponse search(const SearchRequest& request);
  core::yield_instruction_t load_root_action_values(const core::YieldNotificationUnit&,
                                                    ActionValueTensor& action_values);
  const LookupTable* lookup_table() const { return &lookup_table_; }
  const RootInfo* root_info() const { return &root_info_; }

  void end_session() { nn_eval_service_->end_session(); }

  void set_post_visit_func(post_visit_func_t func) { post_visit_func_ = func; }

 private:
  using context_vec_t = std::vector<SearchContext>;
  using context_id_queue_t = std::queue<core::context_id_t>;

  Manager(bool dummy, mutex_vec_sptr_t node_mutex_pool, mutex_vec_sptr_t context_mutex_pool,
          const ManagerParams& params, core::GameServerBase*, NNEvaluationServiceBase* service);

  SearchResponse search_helper(const SearchRequest& request);

  // Assumes state_matchine_.mutex is held
  //
  // If state_machine_.state is already kInVisitLoop, this function does nothing.
  //
  // Otherwise, sets state_machine_.state to kInVisitLoop and does various bookkeeping on
  // context and the other SearchContext's. Returns the extra_enqueue_count value which should be
  // written to the search response.
  int update_state_machine_to_in_visit_loop(SearchContext& context);

  // Assumes state_matchine_.mutex is held
  core::yield_instruction_t mark_as_done_with_visit_loop(SearchContext& context,
                                                         int extra_enqueue_count);

  void init_context(core::context_id_t);
  void init_root_info(bool add_noise);
  bool more_search_iterations_needed(Node* root);
  core::yield_instruction_t begin_root_initialization(SearchContext&);
  core::yield_instruction_t resume_root_initialization(SearchContext&);
  core::yield_instruction_t begin_node_initialization(SearchContext&);
  core::yield_instruction_t resume_node_initialization(SearchContext& context);
  core::yield_instruction_t begin_search_iteration(SearchContext& context);
  core::yield_instruction_t resume_search_iteration(SearchContext& context);
  core::yield_instruction_t begin_visit(SearchContext& context);
  core::yield_instruction_t resume_visit(SearchContext& context);
  core::yield_instruction_t begin_expansion(SearchContext& context);
  core::yield_instruction_t resume_expansion(SearchContext& context);

  void add_pending_notification(SearchContext&, Edge*);
  void set_edge_state(SearchContext&, Edge*, expansion_state_t);
  void transform_policy(node_pool_index_t index, LocalPolicyArray& P) const;
  void add_dirichlet_noise(LocalPolicyArray& P) const;
  void expand_all_children(SearchContext& context, Node* node);
  void virtual_backprop(SearchContext& context);
  void undo_virtual_backprop(SearchContext& context);
  void pure_backprop(SearchContext& context, const ValueArray& value);
  void standard_backprop(SearchContext& context, bool undo_virtual);
  void short_circuit_backprop(SearchContext& context);
  void calc_canonical_state_data(SearchContext& context);
  void print_visit_info(const SearchContext& context);
  void validate_search_path(const SearchContext& context) const;
  int get_best_child_index(const SearchContext& context);
  int sample_chance_child_index(const SearchContext& context);
  std::string search_path_str(const SearchContext& context) const;  // slow, for debugging
  void print_action_selection_details(const SearchContext& context, const ActionSelector& selector,
                                      int argmax_index) const;

  void prepare_results();

  void load_action_symmetries(Node* root, core::action_t* actions);
  void prune_policy_target(const SearchParams&, group::element_t inv_sym);

  static int next_instance_id_;

  const ManagerParams params_;
  const SearchParams pondering_search_params_;
  const int manager_id_ = -1;

  LookupTable lookup_table_;
  mutable eigen_util::UniformDirichletGen<float> dirichlet_gen_;
  math::ExponentialDecay root_softmax_temperature_;
  mutable Eigen::Rand::P8_mt19937_64 rng_;
  RootInfo root_info_;
  post_visit_func_t post_visit_func_;

  context_vec_t contexts_;
  StateMachine state_machine_;
  mutex_vec_sptr_t context_mutex_pool_;
  NNEvaluationServiceBase* nn_eval_service_ = nullptr;

  SearchParams search_params_;
  SearchResults results_;

  bool mid_load_root_action_values_ = false;
  bool connected_ = false;
};

}  // namespace mcts

#include "inline/mcts/Manager.inl"
