#pragma once

#include "core/BasicTypes.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

namespace core {

struct NodeBase {};

// core::Node<StableData, Stats> is the node used in the search framework.
//
// It consists of data and methods that are shared across all different search-frameworks
// (e.g., alpha0 and beta0).
//
// The StableData and Stats classes are specialized for each framework.
//
// It consists of three main parts:
//
// 1. StableData: write-once data that is fixed for the lifetime of the node
// 2. Stats: mutable data that can be updated during the lifetime of the node
// 3. Edge[]: edges to children nodes, needed for tree traversal
//
// The Edge[] array is effectively represented by a an Edge* and an array length. The Edge* is
// captured by first_edge_index_ (an index into a pool of edge objects), and the array length is
// gotten from stable_data_.num_valid_actions.
template <typename StableData, typename Stats>
class Node : public NodeBase {
 public:
  template <typename... Ts>
  Node(mit::mutex* m, Ts&&... args) : mutex_(m), stable_data_(std::forward<Ts>(args)...) {}

  mit::mutex& mutex() const { return *mutex_; }
  int child_expand_count() const { return child_expand_count_; }

  bool is_terminal() const { return stable_data_.terminal; }
  core::action_mode_t action_mode() const { return stable_data_.action_mode; }

  bool edges_initialized() const { return first_edge_index_ != -1; }
  core::edge_pool_index_t get_first_edge_index() const { return first_edge_index_; }
  void set_first_edge_index(core::edge_pool_index_t e) { first_edge_index_ = e; }
  const StableData& stable_data() const { return stable_data_; }
  StableData& stable_data() { return stable_data_; }

  // stats() returns a reference to the stats object, WITHOUT acquiring the mutex. In order to use
  // this function properly, the caller must ensure that one of the following is true:
  //
  // 1. The context is single-threaded, or,
  // 2. The usage of the stats reference falls within the scope of the node's mutex, or,
  // 3. The caller is ok with the possibility of a race-condition with a writer.
  const Stats& stats() const { return stats_; }
  Stats& stats() { return stats_; }

  // Acquires the mutex and returns a copy of the stats object.
  Stats stats_safe() const;

 protected:
  mit::mutex* mutex_;  // potentially shared by multiple Node's

  StableData stable_data_;
  Stats stats_;
  core::edge_pool_index_t first_edge_index_ = -1;

  int child_expand_count_ = 0;
  bool trivial_ = false;  // set to true if all actions discovered to be symmetrically equivalent
};

}  // namespace core

#include "inline/core/Node.inl"
