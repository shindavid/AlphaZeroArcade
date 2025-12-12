#pragma once

#include "alphazero/Edge.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

/*
 * An Edge corresponds to an action that can be taken from this node.
 */
template <core::concepts::EvalSpec EvalSpec>
struct Edge : public alpha0::Edge<EvalSpec> {
  using LogitValueArray = EvalSpec::Game::Types::LogitValueArray;
  using ValueArray = EvalSpec::Game::Types::ValueArray;
  using Base = alpha0::Edge<EvalSpec>;

  // In alpha0, the edge-count *is* the (unnormalized) posterior policy value.
  //
  // In beta0, we decouple the two concepts, and so track the posterior policy value separately.
  float policy_posterior_prob = 0;

  // child_AU is set to with the neural network's AU estimate of this edge's action at the time the
  // parent is evaluated.
  ValueArray child_AU;

  LogitValueArray child_logit_value_beliefs;
};

}  // namespace beta0
