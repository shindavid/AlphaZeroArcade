#pragma once

#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/YieldManager.hpp"
#include "search/GeneralContext.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchContext.hpp"
#include "search/SearchParams.hpp"
#include "search/SearchRequest.hpp"
#include "search/SearchResponse.hpp"
#include "search/TraitsTypes.hpp"
#include "search/TypeDefs.hpp"
#include "util/Math.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <array>
#include <memory>
#include <queue>
#include <vector>

namespace mcts {

/*
 * The Manager class is the main entry point for doing MCTS searches.
 *
 * It maintains the search-tree and manages the threads and services that perform the search.
 */
template <typename Traits>
class Manager {
 public:
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using SearchResults = Traits::SearchResults;
  using ManagerParams = Traits::ManagerParams;
  using Algorithms = Traits::Algorithms;
  using EvalRequest = Traits::EvalRequest;
  using EvalResponse = Traits::EvalResponse;
  using EvalServiceBase = Traits::EvalServiceBase;
  using EvalServiceFactory = Traits::EvalServiceFactory;
  using EvalServiceBase_sptr = std::shared_ptr<EvalServiceBase>;
  using TraitsTypes = search::TraitsTypes<Traits>;
  using Visitation = TraitsTypes::Visitation;

  using LocalPolicyArray = Node::LocalPolicyArray;
  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;
  using ActionValueTensor = Game::Types::ActionValueTensor;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;

  using GeneralContext = search::GeneralContext<Traits>;
  using RootInfo = GeneralContext::RootInfo;
  using LookupTable = search::LookupTable<Traits>;
  using SearchContext = search::SearchContext<Traits>;
  using SearchResponse = search::SearchResponse<Traits>;

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
  using InputTensorizor = Game::InputTensorizor;
  using MCTSKey = InputTensorizor::MCTSKey;
  using StateHistoryArray = std::array<StateHistory, SymmetryGroup::kOrder>;

  using ValueTensor = Game::Types::ValueTensor;
  using ValueArray = Game::Types::ValueArray;

  using post_visit_func_t = std::function<void()>;

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
          EvalServiceBase_sptr service = nullptr);

  Manager(search::mutex_vec_sptr_t& node_mutex_pool, search::mutex_vec_sptr_t& context_mutex_pool,
          const ManagerParams& params, core::GameServerBase* server = nullptr,
          EvalServiceBase_sptr service = nullptr);

  ~Manager();

  const ManagerParams& params() const { return general_context_.manager_params; }
  int num_search_threads() const { return params().num_search_threads; }
  bool multithreaded() const { return num_search_threads() > 1; }

  void start();
  void clear();
  void receive_state_change(core::seat_index_t, const State&, core::action_t);
  void update(core::action_t);

  void set_search_params(const search::SearchParams& search_params);
  SearchResponse search(const search::SearchRequest& request);
  core::yield_instruction_t load_root_action_values(const core::YieldNotificationUnit&,
                                                    ActionValueTensor& action_values);
  const LookupTable* lookup_table() const { return &general_context_.lookup_table; }
  const RootInfo* root_info() const { return &general_context_.root_info; }
  LookupTable* lookup_table() { return &general_context_.lookup_table; }
  RootInfo* root_info() { return &general_context_.root_info; }

  void end_session() { nn_eval_service_->end_session(); }

  void set_post_visit_func(post_visit_func_t func) { post_visit_func_ = func; }

 private:
  using context_vec_t = std::vector<SearchContext>;
  using context_id_queue_t = std::queue<core::context_id_t>;

  Manager(bool dummy, search::mutex_vec_sptr_t node_mutex_pool,
          search::mutex_vec_sptr_t context_mutex_pool, const ManagerParams& params,
          core::GameServerBase*, EvalServiceBase_sptr service);

  SearchResponse search_helper(const search::SearchRequest& request);

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
  void set_edge_state(SearchContext&, Edge*, search::expansion_state_t);
  void transform_policy(search::node_pool_index_t index, LocalPolicyArray& P) const;
  void add_dirichlet_noise(LocalPolicyArray& P) const;
  void expand_all_children(SearchContext& context, Node* node);
  void calc_canonical_state_data(SearchContext& context);
  int sample_chance_child_index(const SearchContext& context);

  void prune_policy_target(group::element_t inv_sym);

  static int next_instance_id_;

  const int manager_id_ = -1;

  GeneralContext general_context_;

  mutable eigen_util::UniformDirichletGen<float> dirichlet_gen_;
  math::ExponentialDecay root_softmax_temperature_;
  mutable Eigen::Rand::P8_mt19937_64 rng_;
  post_visit_func_t post_visit_func_;

  context_vec_t contexts_;
  StateMachine state_machine_;
  search::mutex_vec_sptr_t context_mutex_pool_;
  EvalServiceBase_sptr nn_eval_service_;

  SearchResults results_;

  bool mid_load_root_action_values_ = false;
  bool connected_ = false;
};

}  // namespace mcts

#include "inline/search/Manager.inl"
