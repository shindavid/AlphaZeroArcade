#pragma once

#include "core/BasicTypes.hpp"
#include "core/StableDataBase.hpp"
#include "alpha0/concepts/SpecConcept.hpp"

namespace alpha0 {

// StableData consists of data members of core::NodeBase<EvalSpec> whose values do not change once
// the node is created. The StableData member of core::NodeBase<EvalSpec> is const in spirit, but
// because we have some Node-copying in LookupTable, we cannot make it truly const.
template <alpha0::concepts::Spec EvalSpec>
struct NodeStableData : public core::StableDataBaseImpl<EvalSpec> {
  using Base = core::StableDataBaseImpl<EvalSpec>;
  using Game = EvalSpec::Game;
  using State = Game::State;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using GameOutcome = Game::Types::GameOutcome;
  using ValueArray = GameResultEncoding::ValueArray;

  NodeStableData(const State&, int n_valid_moves, core::seat_index_t);  // for non-terminal nodes
  NodeStableData(const State&, const GameOutcome& game_outcome);        // for terminal nodes

  ValueArray V() const { return GameResultEncoding::to_value_array(R); }

  GameResultEncoding::Tensor R;
  int num_valid_moves;

  // active_seat is usually the current player, who is about to make a move
  // if this is a chance node, active_seat is the player who just made a move
  core::seat_index_t active_seat;

  bool terminal;
  bool R_valid;
  bool is_chance_node;
};

}  // namespace alpha0

#include "inline/alpha0/NodeStableData.inl"
