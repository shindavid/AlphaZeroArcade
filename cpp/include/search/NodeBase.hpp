#pragma once

#include "core/BasicTypes.hpp"
#include "core/NodeBaseCore.hpp"
#include "search/LookupTable.hpp"

namespace search {

// search::NodeBase<Game> is a base class of mcts::Node<Game>.
//
// It consists of data and methods that are shared across all different search-frameworks
// (e.g., MCTS and Bayesian-MCTS)

template <typename Traits>
class NodeBase : public core::NodeBaseCore<typename Traits::Game> {
 public:
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using NodeBaseCore = core::NodeBaseCore<Game>;
  using StateHistory = Game::StateHistory;
  using ActionMask = Game::Types::ActionMask;
  using ValueTensor = Game::Types::ValueTensor;
  using LookupTable = search::LookupTable<Traits>;

  template <typename... Ts>
  NodeBase(LookupTable*, Ts&&... args);

  mit::mutex& mutex() const { return this->lookup_table_->get_mutex(mutex_id_); }

  void initialize_edges();

  Edge* get_edge(int i) const;
  Node* get_child(const Edge* edge) const;
  core::node_pool_index_t lookup_child_by_action(core::action_t action) const;
  void update_child_expand_count(int n = 1);
  bool all_children_edges_initialized() const;
  bool trivial() const { return trivial_; }

 protected:
  LookupTable* lookup_table_;

  // Each Node has an int mutex_id_, rather than an actual mutex. This is for 2 reasons:
  //
  // 1. Allows multiple Node's to share the same mutex
  // 2. Allows for the Node object to copied and moved around (which is needed for defragmentation)
  int mutex_id_;

  int child_expand_count_ = 0;

  bool trivial_ = false;  // set to true if all actions discovered to be symmetrically equivalent
};

}  // namespace search

#include "inline/search/NodeBase.inl"
