#pragma once

#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/SharedData.hpp>
#include <mcts/TreeTraversalThread.hpp>

#include <boost/filesystem.hpp>

#include <mutex>
#include <vector>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class PrefetchThread;

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class PrefetchThreadManager {
 public:
  using thread_vec_t = std::vector<PrefetchThread<GameState, Tensorizor>*>;
  using SharedData = mcts::SharedData<GameState, Tensorizor>;
  using NNEvaluationService = mcts::NNEvaluationService<GameState, Tensorizor>;

  struct work_item_t {
    SharedData* shared_data = nullptr;
    NNEvaluationService* nn_eval_service = nullptr;
    const SearchParams* search_params = nullptr;
    const ManagerParams* manager_params = nullptr;
  };

  /*
   * Gets a singleton PrefetchThreadManager, to be shared by all mcts::Manager's for this
   * (GameState, Tensorizor).
   *
   * The passed-in params has a num_search_threads field that specifies the number of threads to
   * use. If multiple calls specify different values for num_search_threads, the singleton is
   * configured to use the maximum of the specified values.
   */
  static PrefetchThreadManager* get(const ManagerParams& params);

  PrefetchThreadManager(const boost::filesystem::path& profiling_dir)
      : profiling_dir_(profiling_dir) {}

  /*
   * Destroys all threads.
   */
  void shutdown();

  /*
   * Adds a work item to the work queue. Notifies all threads waiting on work_items_cv_.
   */
  void add_work(SharedData*, NNEvaluationService*, const SearchParams*, const ManagerParams*);

  /*
   * Marks the SharedData as not seeking search threads, and removes the matching work item from the
   * work queue. Notifies all threads waiting on work_items_cv_.
   */
  void remove_work(SharedData*);

  /*
   * Waits until the SharedData is no longer seeking search threads. Then waits until no more
   * search threads are working on it.
   */
  void wait_for_completion(SharedData*);

  std::mutex& work_items_mutex() { return work_items_mutex_; }
  std::condition_variable& work_items_cv() { return work_items_cv_; }

  /*
   * If a shutdown has been initiated, returns false.
   *
   * Otherwise, sets *work_item to an available work item and returns true.
   */
  bool get_next_work_item(work_item_t* work_item);

  const boost::filesystem::path& profiling_dir() const { return profiling_dir_; }

  bool shutdown_initiated() const { return shutdown_initiated_; }

 private:
  using work_item_vec_t = std::vector<work_item_t>;

  /*
   * Helper to get_next_work_item(). Assumes work_items_mutex_ is locked.
   */
  bool get_next_work_item_helper(work_item_t* work_item);

  /*
   * Adds more threads until the total number of threads is at least num_total_threads.
   */
  void add_threads_if_necessary(int num_total_threads);

  static PrefetchThreadManager* instance_;

  boost::filesystem::path profiling_dir_;
  thread_vec_t threads_;
  work_item_vec_t work_items_;
  int work_item_index_ = 0;
  bool shutdown_initiated_ = false;

  std::mutex work_items_mutex_;
  std::condition_variable work_items_cv_;
};

/*
 * See documentation for TreeTraversalThread.
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class PrefetchThread : public TreeTraversalThread<GameState, Tensorizor> {
 public:
  using base_t = TreeTraversalThread<GameState, Tensorizor>;

  PrefetchThread(PrefetchThreadManager* manager, int thread_id);

 protected:
  void loop();
  void prefetch(Node* tree, edge_t* edge, move_number_t move_number);
  void virtual_backprop();
  void backprop_with_virtual_undo(const ValueArray& value);

  PrefetchThreadManager* const manager_;
};

}  // namespace mcts

#include <mcts/inl/PrefetchThread.inl>
