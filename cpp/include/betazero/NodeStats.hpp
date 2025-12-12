#pragma once

#include "alphazero/NodeStats.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct NodeStats : public alpha0::NodeStats<EvalSpec> {
  using Base = alpha0::NodeStats<EvalSpec>;
  using ValueArray = Base::ValueArray;
  using LogitValueArray = EvalSpec::Game::Types::LogitValueArray;

  NodeStats();

  // The logit-normal belief about the value of this node. This is a direct function of Q and W,
  // but we store it here for computational savings.
  LogitValueArray logit_value_beliefs;

  ValueArray Q_min;  // min Q observed per player
  ValueArray Q_max;  // for each player, the max Q ever observed for that player
  ValueArray W;      // uncertainty
  ValueArray W_max;  // for each player, the max W ever observed for that player
};

}  // namespace beta0

#include "inline/betazero/NodeStats.inl"
