#pragma once

#include "search/StableData.hpp"
#include "search/TypeDefs.hpp"

namespace search {

// Object hierarchy:
//
// search::NodeBaseCore<Traits>
// └── search::NodeBase<Traits>
//     ├── mcts::Node<Traits::Game>
//     └── bayesian_mcts::Node<Traits::Game>
//
// NodeBase holds members that are common to all node types.
//
// NodeBaseCore is pulled out of NodeBase in order to break a circular dependency between
// NodeBase and LookupTable.

template <typename Traits>
class NodeBaseCore {
 public:
  using StableData = search::StableData<Traits>;

  template <typename... Ts>
  NodeBaseCore(Ts&&... args) : stable_data_(std::forward<Ts>(args)...) {}

  bool is_terminal() const { return stable_data_.terminal; }
  core::action_mode_t action_mode() const { return stable_data_.action_mode; }

  bool edges_initialized() const { return first_edge_index_ != -1; }
  edge_pool_index_t get_first_edge_index() const { return first_edge_index_; }
  void set_first_edge_index(edge_pool_index_t e) { first_edge_index_ = e; }
  const StableData& stable_data() const { return stable_data_; }

 protected:
  StableData stable_data_;
  edge_pool_index_t first_edge_index_ = -1;
};

}  // namespace search
