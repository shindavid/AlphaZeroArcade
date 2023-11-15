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

  void activate_search() {
    std::unique_lock lock(search_mutex_);
    if (search_active_) return;
    search_active_ = true;
    lock.unlock();

    search_begin_cv_.notify_all();
  }

  void deactivate_search() {
    std::unique_lock lock(search_mutex_);
    if (!search_active_) return;
    search_active_ = false;
    lock.unlock();

    search_end_cv_.notify_all();
  }

  bool search_active() const { return search_active_; }
  bool shutdown_initiated() const { return shutdown_initiated_; }

  void increment_active_thread_count() {
    std::unique_lock lock(search_mutex_);
    active_thread_count_++;
  }

  void decrement_active_thread_count() {
    std::unique_lock lock(search_mutex_);
    active_thread_count_--;
    if (active_thread_count_ == 0) {
      lock.unlock();
      search_end_cv_.notify_all();
    }
  }

  void wait_for_search_activation() {
    std::unique_lock lock(search_mutex_);
    search_begin_cv_.wait(lock, [&]() { return search_active_ || shutdown_initiated_; });
  }

  void wait_for_search_completion() {
    std::unique_lock lock(search_mutex_);
    search_end_cv_.wait(lock, [&]() { return !search_active_ && active_thread_count_ == 0; });
  }

  void shutdown() {
    std::unique_lock lock(search_mutex_);
    shutdown_initiated_ = true;
    lock.unlock();
    search_begin_cv_.notify_all();
  }

  /*
   * Notifies that a PrefetchThread has done some work.
   */
  void prefetch_notify() { prefetch_cv_.notify_all(); }

  /*
   * Waits until a PrefetchThread has evaluated node, or until root's visit count has grown by
   * at least root_prefetch_count_limit.
   *
   * Returns true if node is evaluated, and false if not.
   */
  bool wait_for_eval(Node* root, Node* node, int root_prefetch_count_limit) {
    auto state = node->evaluation_data().state;
    if (state == Node::kSet) return true;

    int root_count = root->stats(kPrefetchMode).real_count;
    int root_count_threshold = root_count + root_prefetch_count_limit;

    std::unique_lock lock(prefetch_mutex_);
    prefetch_cv_.wait(lock, [&]() {
      state = node->evaluation_data().state;
      return state == Node::kSet || root->stats(kPrefetchMode).real_count >= root_count_threshold;
    });

    return state == Node::kSet;
  }

  /*
   * Waits until a PrefetchThread has expanded the edge from node for action_index, or until root's
   * visit count has grown by at least root_prefetch_count_limit.
   *
   * Returns the edge if it is expanded, and nullptr if not.
   */
  edge_t* wait_for_edge(Node* root, Node* node, core::action_index_t action_index,
                        int root_prefetch_count_limit) {
    auto& children_data = node->children_data();
    edge_t* edge = children_data.find(action_index);
    if (edge) return edge;

    int root_count = root->stats(kPrefetchMode).real_count;
    int root_count_threshold = root_count + root_prefetch_count_limit;

    std::unique_lock lock(prefetch_mutex_);
    prefetch_cv_.wait(lock, [&]() {
      edge = children_data.find(action_index);
      return edge || root->stats(kPrefetchMode).real_count >= root_count_threshold;
    });

    return edge;
  }

  /*
   * Performs the following actions:
   *
   * 1. Stops all prefetch threads working on this tree.
   * 2. Copies search stats to prefetch stats for all nodes in this tree.
   * 3. Resumes stopped prefetch threads.
   */
  void reset_prefetch_threads() {
    throw util::Exception("reset_prefetch_threads() not implemented");
  }

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
