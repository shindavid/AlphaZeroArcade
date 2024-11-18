#include <mcts/ReachableSet.hpp>

#include <util/Asserts.hpp>

namespace mcts {

template <core::concepts::Game Game>
void ReachableSet<Game>::clear() {
  std::lock_guard lock(mutex_);
  reachable_nodes_.reset();
  reachable_leaves_.reset();
}

template <core::concepts::Game Game>
void ReachableSet<Game>::reset(const LookupTable* table, node_pool_index_t root_index) {
  util::debug_assert(root_index >= 0);
  std::lock_guard lock(mutex_);

  reachable_nodes_.reset();
  reachable_leaves_.reset();

  auto& queue = tmp_deque_;
  queue.clear();
  queue.push_back(root_index);
  reachable_nodes_.set(root_index);

  while (!queue.empty()) {
    node_pool_index_t index = queue.front();
    queue.pop_front();

    const Node* node = table->get_node(index);

    bool has_viable_edge = false;
    if (node->edges_initialized()) {
      int n_edges = node->stable_data().num_valid_actions;
      for (int i = 0; i < n_edges; ++i) {
        const edge_t* edge = node->get_edge(i);
        if (edge->viable()) {
          has_viable_edge = true;

          if (!reachable_nodes_[edge->child_index]) {
            util::debug_assert(edge->child_index >= 0);
            queue.push_back(edge->child_index);
            reachable_nodes_[edge->child_index] = true;
          }
        }
      }
    }

    if (!has_viable_edge) {
      reachable_leaves_.set(index);
    }
  }
}

template <core::concepts::Game Game>
void ReachableSet<Game>::grow(const LookupTable* table, node_pool_index_t parent_index,
                              edge_t* edge) {
  throw std::runtime_error("Not implemented");
}

template <core::concepts::Game Game>
void ReachableSet<Game>::eliminate(edge_t* edge) {
  throw std::runtime_error("Not implemented");
}

template <core::concepts::Game Game>
size_t ReachableSet<Game>::num_reachable_nodes() const {
  return reachable_nodes_.count();
}

template <core::concepts::Game Game>
size_t ReachableSet<Game>::num_reachable_leaves() const {
  return reachable_leaves_.count();
}

}  // namespace mcts
