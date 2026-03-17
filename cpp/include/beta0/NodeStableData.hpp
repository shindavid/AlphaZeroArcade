#pragma once

#include "alpha0/NodeStableData.hpp"
#include "beta0/Calculations.hpp"
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
  using LogitValueArray = EvalSpec::Game::Types::LogitValueArray;
  using ActionMask = Game::Types::ActionMask;

  // non-terminal states - U and lUV will be initialized later
  NodeStableData(const State& state, const ActionMask& mask, core::seat_index_t seat)
      : Base(state, mask, seat) {}

  // terminal states - initialize U and lUV here
  NodeStableData(const State& state, const GameResultTensor& game_outcome)
      : Base(state, game_outcome) {
    U.setZero();
    ValueArray V = Game::GameResults::to_value_array(game_outcome);
    Calculations<Game>::p2l(V, U, lVU);
  }

  ValueArray U;  // uncertainty
  LogitValueArray lVU;
};

}  // namespace beta0
