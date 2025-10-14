#pragma once

#include "alphazero/NodeStats.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct NodeStats : public alpha0::NodeStats<EvalSpec> {
  using Base = alpha0::NodeStats<EvalSpec>;
  using ValueArray = Base::ValueArray;

  NodeStats();
  void init_q(const ValueArray&, bool pure);
  void update_q(const ValueArray&);

  ValueArray Q_min;  // for each player, the minimum value of Q ever observed for that player
  ValueArray Q_max;  // for each player, the maximum value of Q ever observed for that player
  ValueArray W;      // uncertainty
};

}  // namespace beta0

#include "inline/betazero/NodeStats.inl"
