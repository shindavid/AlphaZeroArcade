#pragma once

#include "core/NodeBaseCore.hpp"

namespace search {

// search::NodeBase<Game> is a base class of mcts::Node<Game>.
//
// It consists of data and methods that are shared across all different search-frameworks
// (e.g., MCTS and Bayesian-MCTS)

template <typename Traits>
class NodeBase : public core::NodeBaseCore<typename Traits::Game> {
 public:
  using Game = Traits::Game;
  using NodeBaseCore = core::NodeBaseCore<Game>;

  template <typename... Ts>
  NodeBase(mit::mutex*, Ts&&... args);

  mit::mutex& mutex() const { return *mutex_; }

  int child_expand_count() const { return child_expand_count_; }

  // Increment child_expand_count_ by n. Returns true if n>0 and if all children are now expanded.
  bool increment_child_expand_count(int n);

  void mark_as_trivial() { trivial_ = true; }
  bool trivial() const { return trivial_; }

 protected:
  mit::mutex* mutex_;  // potentially shared by multiple Node's

  int child_expand_count_ = 0;
  bool trivial_ = false;  // set to true if all actions discovered to be symmetrically equivalent
};

}  // namespace search

#include "inline/search/NodeBase.inl"
