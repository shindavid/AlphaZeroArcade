#pragma once

#include <EigenRand/EigenRand>

#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/Node.hpp>
#include <mcts/NodeCache.hpp>
#include <util/EigenUtil.hpp>
#include <util/Math.hpp>

#include <mutex>

namespace mcts {

/*
 * TreeData is owned by the Manager and shared by other threads/services.
 *
 * It is separated from Manager to avoid circular dependencies.
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
struct TreeData {
  using Node = mcts::Node<GameState, Tensorizor>;
  using NodeCache = mcts::NodeCache<GameState, Tensorizor>;
  using edge_t = typename Node::edge_t;

  eigen_util::UniformDirichletGen<float> dirichlet_gen;
  math::ExponentialDecay root_softmax_temperature;
  Eigen::Rand::P8_mt19937_64 rng;

  NodeCache node_cache;
  Node::sptr root_node;
  int manager_id = -1;
  move_number_t move_number = 0;

  /*
   * Signals to TreeTraversalThread's that they should start traversing. Called by Manager.
   */
  void activate_search();

  /*
   * Signals to TreeTraversalThread's that they should stop traversing. Called by the SearchThread
   * when it has reached the tree size limit.
   */
  void deactivate_search();

  bool search_active() const { return search_active_; }
  bool shutdown_initiated() const { return shutdown_initiated_; }

  /*
   * Safetly increments the active thread count, which is used to determine when it is safe for
   * the Manager to modify objects that the TreeTraversalThread's use.
   */
  void increment_active_thread_count();

  /*
   * Safetly decrements the active thread count, which is used to determine when it is safe for
   * the Manager to modify objects that the TreeTraversalThread's use.
   *
   * If this decrement causes the active thread count to reach 0, then the Manager is notified.
   */
  void decrement_active_thread_count();

  /*
   * Waits until either search is activated or shutdown is initiated. Both actions are taken by the
   * Manager.
   *
   * Called by the SearchThread, which either continue's or break's out of its loop depending on
   * which condition is met.
   */
  void wait_for_search_activation();

  /*
   * Waits until search is deactivated and the active thread count is zero. Called by the Manager
   * to make sure it doesn't start modifying objects that are actively used by the
   * TreeTraversalThread's.
   */
  void wait_for_search_completion();

  /*
   * Called at program shutdown by the Manager. Signals to all TreeTraversalThread's that they can
   * exit their loops.
   */
  void shutdown();

  /*
   * Notifies the SearchThread that a PrefetchThread has done some work. Called by PrefetchThread.
   *
   * The SearchThread cares about this because it needs to reset the prefetch threads if they have
   * done a lot of work without the SearchThread making any progress.
   */
  void prefetch_notify() { prefetch_cv_.notify_all(); }

  /*
   * Waits until a PrefetchThread has evaluated node, or until root's visit count has grown by
   * at least root_prefetch_count_limit.
   *
   * Returns true if node is evaluated, and false if not.
   *
   * Called by the SearchThread. The return value is used to determine whether or not to issue
   * a reset command to the prefetch threads.
   */
  bool wait_for_eval(Node* root, Node* node, int root_prefetch_count_limit);

  /*
   * Waits until a PrefetchThread has expanded the edge from node for action_index, or until root's
   * visit count has grown by at least root_prefetch_count_limit.
   *
   * Returns the edge if it is expanded, and nullptr if not.
   *
   * Called by the SearchThread. The return value is used to determine whether or not to issue
   * a reset command to the prefetch threads.
   */
  edge_t* wait_for_edge(Node* root, Node* node, core::action_index_t action_index,
                        int root_prefetch_count_limit);
  /*
   * Performs the following actions:
   *
   * 1. Stops all prefetch threads working on this tree.
   * 2. Copies search stats to prefetch stats for all nodes in this tree.
   * 3. Resumes stopped prefetch threads.
   *
   * Called by the SearchThread if the prefetch threads have done a lot of work without the
   * SearchThread making any progress.
   */
  void reset_prefetch_threads();

 private:
  mutable std::mutex prefetch_mutex_;
  mutable std::mutex search_mutex_;
  std::condition_variable prefetch_cv_;
  std::condition_variable search_begin_cv_;
  std::condition_variable search_end_cv_;
  int active_thread_count_ = 0;
  bool search_active_ = false;
  bool shutdown_initiated_ = false;
};

}  // namespace mcts

#include <mcts/inl/TreeData.inl>
