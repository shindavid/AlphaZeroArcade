#pragma once

#include <mutex>
#include <thread>
#include <vector>

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Node.hpp>

namespace mcts {

/*
 * Deleting a Node can be costly, as it requires traversing the tree to delete all descendents. This service allows you
 * to schedule Node's for deletion in a separate thread, so that the main thread can continue searching without being
 * forced to wait for the deletion to complete.
 */
template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class NodeReleaseService {
public:
  using Node = mcts::Node<GameState, Tensorizor>;

  struct work_unit_t {
    work_unit_t(Node* n, Node* a) : node(n), arg(a) {}

    Node* node;
    Node* arg;
  };

  static void release(Node* node, Node* arg=nullptr) { instance_.release_helper(node, arg); }

private:
  NodeReleaseService();
  ~NodeReleaseService();

  void loop();
  void release_helper(Node* node, Node* arg);

  static NodeReleaseService instance_;

  using work_queue_t = std::vector<work_unit_t>;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  work_queue_t work_queue_[2];
  int queue_index_ = 0;
  int release_count_ = 0;
  int max_queue_size_ = 0;
  bool destructing_ = false;
};

}  // namespace mcts

#include <mcts/inl/NodeReleaseService.inl>
