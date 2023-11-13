#pragma once

#include <EigenRand/EigenRand>

#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/NodeCache.hpp>
#include <util/EigenUtil.hpp>
#include <util/Math.hpp>

#include <mutex>

namespace mcts {

/*
 * SharedData is owned by the Manager and shared by other threads/services.
 *
 * It is separated from Manager to avoid circular dependencies.
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
struct SharedData {
  using Node = mcts::Node<GameState, Tensorizor>;
  using NodeCache = mcts::NodeCache<GameState, Tensorizor>;

  eigen_util::UniformDirichletGen<float> dirichlet_gen;
  math::ExponentialDecay root_softmax_temperature;
  Eigen::Rand::P8_mt19937_64 rng;

  NodeCache node_cache;
  Node::sptr root_node;
  int manager_id = -1;
  move_number_t move_number = 0;

  void activate_search() {
    std::unique_lock lock(mutex_);
    if (search_active_) return;
    search_active_ = true;
    lock.unlock();

    search_begin_cv_.notify_all();
  }

  void deactivate_search() {
    std::unique_lock lock(mutex_);
    if (!search_active_) return;
    search_active_ = false;
    lock.unlock();

    search_end_cv_.notify_all();
  }

  bool search_active() const { return search_active_; }

  void increment_active_thread_count() {
    std::unique_lock lock(mutex_);
    active_thread_count_++;
  }

  void decrement_active_thread_count() {
    std::unique_lock lock(mutex_);
    active_thread_count_--;
    if (active_thread_count_ == 0) {
      lock.unlock();
      search_end_cv_.notify_all();
    }
  }

  void wait_for_search_activation() {
    std::unique_lock lock(mutex_);
    search_begin_cv_.wait(lock, [&]() { return search_active_; });
  }

  void wait_for_search_completion() {
    std::unique_lock lock(mutex_);
    search_end_cv_.wait(lock, [&]() { return !search_active_ && active_thread_count_ == 0; });
  }

private:
  std::mutex mutex_;
  std::condition_variable search_begin_cv_;
  std::condition_variable search_end_cv_;
  int active_thread_count_ = 0;
  bool search_active_ = false;
};

}  // namespace mcts
