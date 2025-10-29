#pragma once

#include "alphazero/NodeStats.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct NodeStats : public alpha0::NodeStats<EvalSpec> {
  using Base = alpha0::NodeStats<EvalSpec>;
  using ValueArray = Base::ValueArray;

  NodeStats();

  // We will eventually remove Qbeta in favor of just Q. We keep it here so that we can track Q
  // identically to alpha0 for now, so that we can reuse alpha0's selection criterion. Once we feel
  // comfortable implementing a new selection criterion for beta0, we can remove Qbeta and change
  // all Qbeta references to Q.
  ValueArray Qbeta;

  ValueArray Qbeta_min;  // for each player, the min Qbeta ever observed for that player
  ValueArray Qbeta_max;  // for each player, the max Qbeta ever observed for that player
  ValueArray W;          // uncertainty
  ValueArray W_max;      // for each player, the max W ever observed for that player
};

}  // namespace beta0

#include "inline/betazero/NodeStats.inl"
