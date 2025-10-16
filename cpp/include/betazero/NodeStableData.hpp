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
  using State = Game::State;
  using Base = alpha0::NodeStableData<EvalSpec>;
  using ValueArray = Game::Types::ValueArray;

  template <typename... Ts>
  NodeStableData(Ts&&... args) : Base(std::forward<Ts>(args)...) {
    U.setZero();
  }

  ValueArray U;  // uncertainty
};

}  // namespace beta0
