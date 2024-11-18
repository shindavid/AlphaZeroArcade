#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/Node.hpp>
#include <util/InfiniteBitset.hpp>

#include <mutex>
#include <deque>

namespace mcts {

/*
 * Tracks the set of all nodes that are reachable from the root of a given mcts tree via "viable"
 * edges of the tree. A viable edge is one that has not been eliminated and which has E > 0.
 */
template <core::concepts::Game Game>
class ReachableSet {
 public:
  using Node = mcts::Node<Game>;
  using node_pool_index_t = Node::node_pool_index_t;
  using LookupTable = Node::LookupTable;
  using edge_t = Node::edge_t;
  using InfiniteBitset = util::InfiniteBitset;

  void clear();

  void reset(const LookupTable* table, node_pool_index_t root_index);

  void grow(const LookupTable* table, node_pool_index_t parent_index, edge_t* edge);

  void eliminate(edge_t* edge);

  size_t num_reachable_nodes() const;
  size_t num_reachable_leaves() const;

 private:
  mutable std::mutex mutex_;
  InfiniteBitset reachable_nodes_;
  InfiniteBitset reachable_leaves_;  // subset of reachable_nodes_

  // The tmp_* members are used for temporary storage. They are used as if they are local variables,
  // but are members to avoid repeated allocation/deallocation.
  std::deque<node_pool_index_t> tmp_deque_;
};

}  // namespace mcts

#include <inline/mcts/ReachableSet.inl>
