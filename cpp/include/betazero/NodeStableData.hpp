#pragma once

#include "alphazero/NodeStableData.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

// StableData consists of data members of core::NodeBase<EvalSpec> whose values do not change once
// the node is created. The StableData member of core::NodeBase<EvalSpec> is const in spirit, but
// because we have some Node-copying in LookupTable, we cannot make it truly const.
template <core::concepts::EvalSpec EvalSpec>
struct NodeStableData : public alpha0::NodeStableData<EvalSpec> {
  using Game = EvalSpec::Game;
  using Base = alpha0::NodeStableData<EvalSpec>;
  using Base::Base;

  // TODO: U should technically be per-player.
  //
  // For 2-player games, we know that the two players will have the same uncertainty, since when
  // you square the difference of two length-2 vectors that individually sum to 1, both entries
  // will be the same.
  //
  // When we have more than 2 players, we will need to revisit this.
  static_assert(Game::Constants::kNumPlayers == 2,
                "beta0::NodeStableData only supports 2-player games for now");

  float U = 0;  // uncertainty
};

}  // namespace beta0
