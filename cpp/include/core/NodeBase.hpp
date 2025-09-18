#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace core {

// core::NodeBase<EvalSpec> is a base class of core::Node<EvalSpec, ...>
//
// It is pulled out so that some nnet::NNEval* classes can operate on Node pointers without
// needing to know the specific Node implementation.
//
// It consists of data and methods that are shared across all different search-frameworks
// (e.g., MCTS and Bayesian-MCTS)
//
// It consists of two main parts:
//
// 1. StableData: write-once data that is fixed for the lifetime of the node
// 2. Edge[]: edges to children nodes, needed for tree traversal
//
// The Edge[] array is effectively represented by a an Edge* and an array length. The Edge* is
// captured by first_edge_index_ (an index into a pool of edge objects), and the array length is
// gotten from stable_data_.num_valid_actions.
template <core::concepts::EvalSpec EvalSpec>
class NodeBase {
 public:
  NodeBase(mit::mutex* m) : mutex_(m) {}

  mit::mutex& mutex() const { return *mutex_; }
  int child_expand_count() const { return child_expand_count_; }

  // Increment child_expand_count_ by n. Returns true if n>0 and if all children are now expanded.
  bool increment_child_expand_count(int n);

  void mark_as_trivial() { trivial_ = true; }
  bool trivial() const { return trivial_; }

  bool edges_initialized() const { return first_edge_index_ != -1; }
  core::edge_pool_index_t get_first_edge_index() const { return first_edge_index_; }
  void set_first_edge_index(core::edge_pool_index_t e) { first_edge_index_ = e; }

 protected:
  mit::mutex* mutex_;  // potentially shared by multiple Node's

  core::edge_pool_index_t first_edge_index_ = -1;

  int child_expand_count_ = 0;
  bool trivial_ = false;  // set to true if all actions discovered to be symmetrically equivalent
};

}  // namespace core

#include "inline/core/NodeBase.inl"
