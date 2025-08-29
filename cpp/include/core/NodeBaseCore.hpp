#pragma once

#include "core/BasicTypes.hpp"
#include "core/StableData.hpp"
#include "core/concepts/Game.hpp"

namespace core {

// Object hierarchy:
//
// core::NodeBaseCore<Traits::Game>
// └── search::NodeBase<Traits>
//     ├── mcts::Node<Traits>
//     └── bayesian_mcts::Node<Traits>
//
// NodeBase holds members that are common to all node types.
//
// NodeBaseCore is pulled out of NodeBase in order to break a circular dependency between
// NodeBase and LookupTable.

template <core::concepts::Game Game>
class NodeBaseCore {
 public:
  using StableData = core::StableData<Game>;

  template <typename... Ts>
  NodeBaseCore(Ts&&... args) : stable_data_(std::forward<Ts>(args)...) {}

  bool is_terminal() const { return stable_data_.terminal; }
  core::action_mode_t action_mode() const { return stable_data_.action_mode; }

  bool edges_initialized() const { return first_edge_index_ != -1; }
  core::edge_pool_index_t get_first_edge_index() const { return first_edge_index_; }
  void set_first_edge_index(core::edge_pool_index_t e) { first_edge_index_ = e; }
  const StableData& stable_data() const { return stable_data_; }

 protected:
  StableData stable_data_;
  core::edge_pool_index_t first_edge_index_ = -1;
};

}  // namespace core
