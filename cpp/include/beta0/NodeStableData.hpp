#pragma once

#include "alpha0/NodeStableData.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

// StableData consists of data members of core::NodeBase<EvalSpec> whose values do not change once
// the node is created. The StableData member of core::NodeBase<EvalSpec> is const in spirit, but
// because we have some Node-copying in LookupTable, we cannot make it truly const.
template <core::concepts::EvalSpec EvalSpec>
struct NodeStableData : public alpha0::NodeStableData<EvalSpec> {
  using Game = EvalSpec::Game;
  using State = Game::State;
  using Base = alpha0::NodeStableData<EvalSpec>;
  using ValueArray = Game::Types::ValueArray;
  using GameResultTensor = Game::Types::GameResultTensor;

  // non-terminal states
  NodeStableData(const State& state, int n_valid_actions, core::seat_index_t seat)
      : Base(state, n_valid_actions, seat) {}

  // terminal states
  NodeStableData(const State& state, const GameResultTensor& game_outcome)
      : Base(state, game_outcome) {
    U.setZero();
  }

  ValueArray U;  // uncertainty
};

}  // namespace beta0
