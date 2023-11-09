#pragma once

#include <bitset>
#include <mutex>
#include <thread>
#include <vector>

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/Node.hpp>
#include <mcts/NodeCache.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/SharedData.hpp>
#include <mcts/TypeDefs.hpp>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class SearchThread;
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class SearchThreadManager {
 public:
  using thread_vec_t = std::vector<SearchThread<GameState, Tensorizor>*>;
  using SharedData = mcts::SharedData<GameState, Tensorizor>;
  using NNEvaluationService = mcts::NNEvaluationService<GameState, Tensorizor>;

  struct work_item_t {
    SharedData* shared_data = nullptr;
    NNEvaluationService* nn_eval_service = nullptr;
    const SearchParams* search_params = nullptr;
    const ManagerParams* manager_params = nullptr;
  };

  /*
   * Gets a singleton SearchThreadManager, to be shared by all mcts::Manager's for this
   * (GameState, Tensorizor).
   *
   * The passed-in params has a num_search_threads field that specifies the number of threads to
   * use. If multiple calls specify different values for num_search_threads, the singleton is
   * configured to use the maximum of the specified values.
   */
  static SearchThreadManager* get(const ManagerParams& params);

  SearchThreadManager(const boost::filesystem::path& profiling_dir)
      : profiling_dir_(profiling_dir) {}

  /*
   * Destroys all threads.
   */
  void shutdown();

  /*
   * Marks the SharedData as seeking search threads, and adds a work item to the work queue.
   * Notifies all threads waiting on cv_.
   */
  void add_work(SharedData*, NNEvaluationService*, const SearchParams*, const ManagerParams*);

  /*
   * Marks the SharedData as not seeking search threads, and removes the matching work item from the
   * work queue. Notifies all threads waiting on cv_.
   */
  void remove_work(SharedData*);  // assumes mutex_ is locked

  /*
   * Waits until the SharedData is no longer seeking search threads. Then waits until no more
   * search threads are working on it.
   */
  void wait_for_completion(SharedData*);

  std::mutex& mutex() { return mutex_; }
  std::condition_variable& cv() { return cv_; }

  /*
   * If there is an available work item, sets *work_item to that work item and returns true. If
   * shutdown has been initiated, returns true.
   *
   * Else, return false.
   *
   * Assumes that the caller has locked mutex_.
   */
  bool get_next_work_item(work_item_t* work_item);

  const boost::filesystem::path& profiling_dir() const { return profiling_dir_; }

  bool shutdown_initiated() const { return shutdown_initiated_; }

 private:
  using work_item_vec_t = std::vector<work_item_t>;

  /*
   * Adds more threads until the total number of threads is at least num_total_threads.
   */
  void add_threads_if_necessary(int num_total_threads);

  static SearchThreadManager* instance_;

  boost::filesystem::path profiling_dir_;
  thread_vec_t threads_;
  work_item_vec_t work_items_;
  int work_item_index_ = 0;
  bool shutdown_initiated_ = false;

  std::mutex mutex_;
  std::condition_variable cv_;
};

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class SearchThread {
 public:
  using GameStateTypes = core::GameStateTypes<GameState>;
  using NNEvaluation = mcts::NNEvaluation<GameState>;
  using NNEvaluationService = mcts::NNEvaluationService<GameState, Tensorizor>;
  using Node = mcts::Node<GameState, Tensorizor>;
  using NodeCache = mcts::NodeCache<GameState, Tensorizor>;
  using PUCTStats = mcts::PUCTStats<GameState, Tensorizor>;
  using SharedData = mcts::SharedData<GameState, Tensorizor>;
  using edge_t = typename Node::edge_t;

  using Action = typename GameStateTypes::Action;
  using ActionMask = typename GameStateTypes::ActionMask;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using NNEvaluation_sptr = typename NNEvaluation::sptr;
  using PolicyShape = typename GameStateTypes::PolicyShape;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueArray = typename GameStateTypes::ValueArray;
  using ValueTensor = typename GameStateTypes::ValueTensor;

  using SearchThreadManager = mcts::SearchThreadManager<GameState, Tensorizor>;
  using work_item_t = typename SearchThreadManager::work_item_t;

  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActionsBound = GameStateTypes::kNumGlobalActionsBound;

  using dtype = torch_util::dtype;
  using profiler_t = search_thread_profiler_t;

  SearchThread(SearchThreadManager* manager, int thread_id);
  ~SearchThread();

  int thread_id() const { return thread_id_; }

  void join();
  void kill();
  void loop();
  bool is_pondering() const { return search_params_->ponder; }

  void dump_profiling_stats() { profiler_.dump(64); }

 private:
  struct VirtualIncrement {
    void operator()(Node* node) const { node->stats().virtual_increment(); }
  };

  struct RealIncrement {
    void operator()(Node* node) const { node->stats().real_increment(); }
  };

  struct IncrementTransfer {
    void operator()(Node* node) const { node->stats().increment_transfer(); }
  };

  struct SetEvalExact {
    SetEvalExact(const ValueArray& value) : value(value) {}
    void operator()(Node* node) const { node->stats().set_eval_exact(value); }
    const ValueArray& value;
  };

  struct SetEvalWithVirtualUndo {
    SetEvalWithVirtualUndo(const ValueArray& value) : value(value) {}
    void operator()(Node* node) const { node->stats().set_eval_with_virtual_undo(value); }
    const ValueArray& value;
  };

  struct evaluation_result_t {
    NNEvaluation_sptr evaluation;
    bool backpropagated_virtual_loss;
  };

  struct visitation_t {
    visitation_t(Node* n, edge_t* e) : node(n), edge(e) {}
    Node* node;
    edge_t* edge;
  };
  using search_path_t = std::vector<visitation_t>;

  void visit(Node* tree, edge_t* edge, move_number_t move_number);
  void add_dirichlet_noise(LocalPolicyArray& P);
  void virtual_backprop();
  void pure_backprop(const ValueArray& value);
  void backprop_with_virtual_undo(const ValueArray& value);
  void short_circuit_backprop(edge_t* last_edge);
  evaluation_result_t evaluate(Node* tree);
  void evaluate_unset(Node* tree, std::unique_lock<std::mutex>* lock, evaluation_result_t* data);
  std::string search_path_str() const;  // slow, for debugging

  /*
   * Used in visit().
   *
   * Applies PUCT criterion to select the best child-index to visit from the given Node.
   *
   * TODO: as we experiment with things like auxiliary NN output heads, dynamic cPUCT values,
   * etc., this method will evolve. It probably makes sense to have the behavior as part of the
   * Tensorizor, since there is coupling with NN architecture (in the form of output heads).
   */
  core::action_index_t get_best_action_index(Node* tree, NNEvaluation* evaluation);

  auto& dirichlet_gen() { return shared_data_->dirichlet_gen; }
  auto& rng() { return shared_data_->rng; }
  float root_softmax_temperature() const { return shared_data_->root_softmax_temperature.value(); }

  SharedData* shared_data_ = nullptr;
  NNEvaluationService* nn_eval_service_ = nullptr;
  const SearchParams* search_params_ = nullptr;
  const ManagerParams* manager_params_ = nullptr;

  search_path_t search_path_;
  profiler_t profiler_;

  SearchThreadManager* const manager_;
  const int thread_id_;
  std::thread* thread_ = nullptr;
};

}  // namespace mcts

#include <mcts/inl/SearchThread.inl>
