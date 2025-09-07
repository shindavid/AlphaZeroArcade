#pragma once

#include "core/BasicTypes.hpp"
#include "core/StableData.hpp"
#include "core/concepts/Game.hpp"

namespace core {

// core::NodeBase<Game> is a base class of alpha0::Node<Game>.
//
// It consists of data and methods that are shared across all different search-frameworks
// (e.g., MCTS and Bayesian-MCTS)

template <core::concepts::Game Game>
class NodeBase {
 public:
  using StableData = core::StableData<Game>;

  template <typename... Ts>
  NodeBase(mit::mutex* m, Ts&&... args) : mutex_(m), stable_data_(std::forward<Ts>(args)...) {}

  mit::mutex& mutex() const { return *mutex_; }
  int child_expand_count() const { return child_expand_count_; }

  // Increment child_expand_count_ by n. Returns true if n>0 and if all children are now expanded.
  bool increment_child_expand_count(int n);

  void mark_as_trivial() { trivial_ = true; }
  bool trivial() const { return trivial_; }

  bool is_terminal() const { return stable_data_.terminal; }
  core::action_mode_t action_mode() const { return stable_data_.action_mode; }

  bool edges_initialized() const { return first_edge_index_ != -1; }
  core::edge_pool_index_t get_first_edge_index() const { return first_edge_index_; }
  void set_first_edge_index(core::edge_pool_index_t e) { first_edge_index_ = e; }
  const StableData& stable_data() const { return stable_data_; }
  StableData& stable_data() { return stable_data_; }

 protected:
  mit::mutex* mutex_;  // potentially shared by multiple Node's

  StableData stable_data_;
  core::edge_pool_index_t first_edge_index_ = -1;

  int child_expand_count_ = 0;
  bool trivial_ = false;  // set to true if all actions discovered to be symmetrically equivalent
};

}  // namespace core

#include "inline/core/NodeBase.inl"
