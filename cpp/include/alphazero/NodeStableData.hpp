#pragma once

#include "core/StableDataBase.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace alpha0 {

// StableData consists of data members of core::NodeBase<EvalSpec> whose values do not change once
// the node is created. The StableData member of core::NodeBase<EvalSpec> is const in spirit, but
// because we have some Node-copying in LookupTable, we cannot make it truly const.
template <core::concepts::EvalSpec EvalSpec>
struct NodeStableData : public core::StableDataBaseImpl<EvalSpec> {
  using Base = core::StableDataBaseImpl<EvalSpec>;
  using Game = EvalSpec::Game;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ValueTensor = Game::Types::ValueTensor;

  NodeStableData(const State&, core::seat_index_t active_seat);   // for non-terminal nodes
  NodeStableData(const State&, const ValueTensor& game_outcome);  // for terminal nodes

  ValueTensor VT;
  ActionMask valid_action_mask;
  int num_valid_actions;
  core::action_mode_t action_mode;

  // active_seat is usually the current player, who is about to make a move
  // if this is a chance node, active_seat is the player who just made a move
  core::seat_index_t active_seat;

  bool terminal;
  bool VT_valid;
  bool is_chance_node;
};

}  // namespace alpha0

#include "inline/alphazero/NodeStableData.inl"
