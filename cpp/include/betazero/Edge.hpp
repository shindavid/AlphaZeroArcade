#pragma once

#include "alphazero/Edge.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

/*
 * An Edge corresponds to an action that can be taken from this node.
 */
template <core::concepts::EvalSpec EvalSpec>
struct Edge : public alpha0::Edge<EvalSpec> {
  using ValueArray = EvalSpec::Game::Types::ValueArray;
  using Base = alpha0::Edge<EvalSpec>;

  Edge() : Base() { child_U_estimate.fill(0); }

  // In alpha0, the edge-count *is* the (unnormalized) posterior policy value.
  //
  // In beta0, we decouple the two concepts, and so track the posterior policy value separately.
  float policy_posterior_prob = 0;

  ValueArray child_U_estimate;  // network estimate of child-value-uncertainty
};

}  // namespace beta0
