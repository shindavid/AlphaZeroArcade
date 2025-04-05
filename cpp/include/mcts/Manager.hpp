#pragma once

#include <condition_variable>
#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <mcts/ActionSelector.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluationRequest.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/NNEvaluationServiceBase.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/TypeDefs.hpp>
#include <util/Math.hpp>

#include <array>
#include <mutex>
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
  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;
  using ActionValueTensor = Game::Types::ActionValueTensor;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;

  using ActionSelector = mcts::ActionSelector<Game>;
  using ChanceDistribution = Game::Types::ChanceDistribution;
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

  struct SearchContext {
    core::search_context_id_t id;

    search_path_t search_path;
    mutable search_thread_profiler_t profiler;

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
    core::nn_evaluation_sequence_id_t nn_eval_seq_id;
    node_pool_index_t node_index_under_initialization = -1;
    node_pool_index_t inserted_node_index = -1;
    bool expanded_new_node = false;

    // visit yield info
    Node* visit_node;
    Edge* visit_edge;
    StateHistory* history;
    group::element_t inv_canonical_sym;
    bool applied_action = false;
  };

  enum execution_state_t : int8_t {
    kIdle,
    kInitializingRoot,
    kInVisitLoop
  };

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
  // Each incoming thread is assigned a SearchContext from a round-robin list of SearchContext's
  // and checks the execution state to determine what to do. Here is a summary:
  //
  // ** kIdle
  //
  // This is the entry point. At program start, SearchContext 0 is marked as the primary context.
  // The first thread to call search() when the state is kIdle should use SearchContext 0, and so
  // recognize that it is the primary context. After transitioning to kInitializingRoot and
  // releasing the mutex, it does the work of intializing the root. This could require a nn-eval
  // (when AV-targets are needed at the root), which can cause a yield. If it does not yield, it
  // completes root initialization and then, after re-acquiring the mutex, transitions to
  // kInVisitLoop.
  //
  // If a thread using a SearchContext besides the primary context calls search() while in this
  // state, it blocks until the state is no longer kIdle. See kInVisitLoop details to understand
  // how this can happen.
  //
  // ** kInitializingRoot
  //
  // If a thread calls search() while in this state, it checks to see if it is the primary context.
  // If so, that means that this context previously started the root initialization work and decided
  // to yield. In this case, we block until the nn-eval is done (during self-play, assuming the
  // GameServer is appropriately configured, this should typically already be done). Then, it
  // transitions to kInVisitLoop. All of this is done without releasing the mutex.
  //
  // If a thread calls search() while in this state and it is *not* the primary context, that means
  // that another thread is currently working on root initialization. In this case, it immediately
  // yields. Note that this same context should not hit this same state twice in a row, since the
  // round-robin assignment of SearchContext's should ensure that the primary context will get
  // there first, and since in that case the primary context will transition the state to
  // kInvisitLoop without releasing the mutex.
  //
  // ** kInVisitLoop
  //
  // This is the main work of the search. The thread that first updates the state to kInVisitLoop
  // will also set visit_loop_count to C.
  //
  // When a thread calls search() while in this state, it immediately releases the mutex, and starts
  // the main work.
  //
  // On the first call, it starts an MCTS iteration. If it reaches a leaf node where it requires an
  // nn-eval, it yields. Otherwise, it completes the MCTS iteration and repeats. Eventually, it
  // should either get to a yield point or hit the termination condition.
  //
  // If it hits a yield point, we record this in the SearchContext, so that on the subsequent visit,
  // it can resume from the yield point. This should be the case for all visits in the kInVisitLoop
  // state except for the first. When resuming, we need to block on the nn-eval to complete: again,
  // this should typically already be done during self-play assuming appropriate GameServer
  // configuration.
  //
  // If it hits the termination condition, it acquires the mutex and decrements visit_loop_count.
  // If visit_loop_count is 0, then this thread knows that none of the SearchContext's are
  // mid-visit, and so it assumes the responsibility of returning results back to the caller (as
  // opposed to yielding, as every other call to search() has done so far). It meets this
  // responsibility by transitioning the state to kIdle, marking its SearchContext as
  // the primary context, and then returning the results, all while holding the mutex.
  //
  // When cycling back to kIdle, note that our assignment of primary context ensures that the
  // thread that returned results is the one that will handle the next kIdle state. This details
  // ensures that we don't have a situation where thread-2 re-enters the visit loop while thread-1
  // is still processing the search results.
  struct StateMachine {
    mutable std::mutex mutex;
    std::condition_variable cv;
    core::search_context_id_t next_context_id = 0;  // round-robins
    core::search_context_id_t primary_context_id = -1;
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
   * Can optionally pass a mutex_cv_pool to be used by the nodes. If not provided, the Manager will
   * create a separate single-element mutex-pool.
   */
  Manager(const ManagerParams& params, NNEvaluationServiceBase* service = nullptr);
  Manager(mutex_cv_vec_sptr_t& mutex_cv_pool, const ManagerParams& params,
          NNEvaluationServiceBase* service = nullptr);

  ~Manager();

  const ManagerParams& params() const { return params_; }
  int num_search_threads() const { return params().num_search_threads; }
  bool multithreaded() const { return num_search_threads() > 1; }

  void start();
  void clear();
  void receive_state_change(core::seat_index_t, const State&, core::action_t);

  void set_search_params(const SearchParams& search_params);
  const SearchResults* search();
  void load_root_action_values(ActionValueTensor& action_values);

  void end_session() { nn_eval_service_->end_session(); }

  void set_post_visit_func(post_visit_func_t func) { post_visit_func_ = func; }

 private:
  using search_context_vec_t = std::vector<SearchContext>;
  using search_context_id_queue_t = std::queue<core::search_context_id_t>;

  Manager(bool dummy, mutex_cv_vec_sptr_t mutex_cv_pool, const ManagerParams& params,
          NNEvaluationServiceBase* service);

  // Assumes state_matchine_.mutex is held
  core::search_context_id_t get_next_context_id();

  // Assumes state_matchine_.mutex is held
  void update_state_machine_to_in_visit_loop();

  // Assumes state_matchine_.mutex is held
  // Returns true if all threads are done with the visit loop
  bool mark_as_done_with_visit_loop(SearchContext& context);

  void init_context(core::search_context_id_t);
  void init_root_info(bool add_noise);
  core::yield_instruction_t begin_root_initialization(SearchContext&);
  void resume_root_initialization(SearchContext&);
  core::yield_instruction_t begin_node_initialization(SearchContext&);
  void resume_node_initialization(SearchContext& context);
  bool more_search_iterations_needed(Node* root);
  core::yield_instruction_t begin_search_iteration(SearchContext& context);
  core::yield_instruction_t resume_search_iteration(SearchContext& context);
  core::yield_instruction_t begin_visit(SearchContext& context);
  void resume_visit(SearchContext& context);
  core::yield_instruction_t begin_expansion(SearchContext& context);
  void resume_expansion(SearchContext& context);

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
  std::string thread_id_whitespace(const SearchContext& context) const;
  std::string break_plus_thread_id_whitespace(const SearchContext& context) const;
  void validate_search_path(const SearchContext& context) const;
  int get_best_child_index(const SearchContext& context);
  int sample_chance_child_index(const SearchContext& context);
  std::string search_path_str(const SearchContext& context) const;  // slow, for debugging
  void print_action_selection_details(const SearchContext& context, const ActionSelector& selector,
                                      int argmax_index) const;

  void prepare_results();

  void announce_shutdown();
  void load_action_symmetries(Node* root, core::action_t* actions);
  void prune_policy_target(const SearchParams&, group::element_t inv_sym);
  static void init_profiling_dir(const std::string& profiling_dir);

  static int next_instance_id_;  // for naming debug/profiling output files

  const ManagerParams params_;
  const SearchParams pondering_search_params_;
  const int manager_id_ = -1;

  LookupTable lookup_table_;
  mutable eigen_util::UniformDirichletGen<float> dirichlet_gen_;
  math::ExponentialDecay root_softmax_temperature_;
  mutable Eigen::Rand::P8_mt19937_64 rng_;
  RootInfo root_info_;
  post_visit_func_t post_visit_func_ = []() {};

  search_context_vec_t search_contexts_;
  StateMachine state_machine_;
  NNEvaluationServiceBase* nn_eval_service_ = nullptr;

  SearchParams search_params_;
  SearchResults results_;

  bool connected_ = false;
};

}  // namespace mcts

#include <inline/mcts/Manager.inl>
