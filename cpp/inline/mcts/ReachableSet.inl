#include <mcts/ReachableSet.hpp>

#include <util/Asserts.hpp>

namespace mcts {

template <core::concepts::Game Game>
void ReachableSet<Game>::clear() {
  std::lock_guard lock(mutex_);
  reachable_nodes_.reset();
  count_ = 0;
}

template <core::concepts::Game Game>
void ReachableSet<Game>::reset(const LookupTable* table, node_pool_index_t root_index) {
  util::debug_assert(root_index >= 0);
  std::lock_guard lock(mutex_);

  reachable_nodes_.reset();

  auto& queue = tmp_deque_;
  queue.clear();
  queue.push_back(root_index);
  reachable_nodes_.set(root_index);

  while (!queue.empty()) {
    node_pool_index_t index = queue.front();
    queue.pop_front();

    const Node* node = table->get_node(index);

    if (!node->edges_initialized()) continue;
    int n_edges = node->stable_data().num_valid_actions;
    for (int i = 0; i < n_edges; ++i) {
      const edge_t* edge = node->get_edge(i);
      if (!edge->viable()) continue;
      if (reachable_nodes_[edge->child_index]) continue;

      queue.push_back(edge->child_index);
      reachable_nodes_[edge->child_index] = true;
    }
  }
  count_ = reachable_nodes_.count();
}

template <core::concepts::Game Game>
void ReachableSet<Game>::add(const LookupTable* table,
                             const std::vector<node_pool_index_t>& indices) {
  std::unique_lock lock(mutex_);
  for (node_pool_index_t node_index : indices) {
    reachable_nodes_.set(node_index);
  }
  count_ = reachable_nodes_.count();
}

}  // namespace mcts
