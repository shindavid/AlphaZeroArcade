#pragma once

#include "alphazero/NodeStats.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct NodeStats : public alpha0::NodeStats<EvalSpec> {
  using Base = alpha0::NodeStats<EvalSpec>;
  using ValueArray = Base::ValueArray;

  void init_q(const ValueArray&, bool pure);

  ValueArray Q_min;  // for each player, the minimum value of Q ever observed for that player
  ValueArray Q_max;  // for each player, the maximum value of Q ever observed for that player
  float W = 0;  // uncertainty
};

}  // namespace beta0

#include "inline/betazero/NodeStats.inl"
