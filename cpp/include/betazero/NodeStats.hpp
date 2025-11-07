#pragma once

#include "alphazero/NodeStats.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct NodeStats : public alpha0::NodeStats<EvalSpec> {
  using Base = alpha0::NodeStats<EvalSpec>;
  using ValueArray = Base::ValueArray;

  NodeStats();

  ValueArray Q_min;  // for each player, the min Q ever observed for that player
  ValueArray Q_max;  // for each player, the max Q ever observed for that player
  ValueArray W;      // uncertainty
  ValueArray W_max;  // for each player, the max W ever observed for that player
};

}  // namespace beta0

#include "inline/betazero/NodeStats.inl"
