#include <mcts/NodeReleaseService.hpp>

namespace mcts {

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
NodeReleaseService<GameState, Tensorizor> NodeReleaseService<GameState, Tensorizor>::instance_;

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
NodeReleaseService<GameState, Tensorizor>::NodeReleaseService()
    : thread_([&] { loop();})
{
  struct sched_param param;
  param.sched_priority = 0;
  pthread_setschedparam(thread_.native_handle(), SCHED_IDLE, &param);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
NodeReleaseService<GameState, Tensorizor>::~NodeReleaseService() {
  destructing_ = true;
  cv_.notify_one();
  if (thread_.joinable()) thread_.join();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NodeReleaseService<GameState, Tensorizor>::loop() {
  while (!destructing_) {
    std::unique_lock<std::mutex> lock(mutex_);
    work_queue_t& queue = work_queue_[queue_index_];
    cv_.wait(lock, [&]{ return !queue.empty() || destructing_;});
    if (destructing_) return;
    queue_index_ = 1 - queue_index_;
    lock.unlock();
    for (auto& unit : queue) {
      unit.node->release(unit.arg);
    }
    queue.clear();
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NodeReleaseService<GameState, Tensorizor>::release_helper(Node* node, Node* arg) {
  std::unique_lock<std::mutex> lock(mutex_);
  work_queue_t& queue = work_queue_[queue_index_];
  queue.emplace_back(node, arg);
  max_queue_size_ = std::max(max_queue_size_, int(queue.size()));
  lock.unlock();
  cv_.notify_one();
  release_count_++;
}

}  // namespace mcts
