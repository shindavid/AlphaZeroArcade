#pragma once

#include "alphazero/NodeStats.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace gamma0 {

template <core::concepts::EvalSpec EvalSpec>
struct NodeStats : public alpha0::NodeStats<EvalSpec> {
  using Base = alpha0::NodeStats<EvalSpec>;
  using ValueArray = Base::ValueArray;

  NodeStats();

  // We will eventually remove Qgamma in favor of just Q. We keep it here so that we can track Q
  // identically to alpha0 for now, so that we can reuse alpha0's selection criterion. Once we feel
  // comfortable implementing a new selection criterion for gamma0, we can remove Qgamma and change
  // all Qgamma references to Q.
  ValueArray Qgamma;

  ValueArray Qgamma_min;  // for each player, the min Qgamma ever observed for that player
  ValueArray Qgamma_max;  // for each player, the max Qgamma ever observed for that player
  ValueArray W;           // uncertainty
  ValueArray W_max;       // for each player, the max W ever observed for that player
};

}  // namespace gamma0

#include "inline/gammazero/NodeStats.inl"
