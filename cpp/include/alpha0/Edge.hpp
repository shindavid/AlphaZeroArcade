#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "search/EdgeBase.hpp"

namespace alpha0 {

/*
 * An Edge corresponds to an action that can be taken from this node.
 */
template <core::concepts::EvalSpec EvalSpec>
struct Edge : public search::EdgeBase {
  using ValueArray = EvalSpec::Game::Types::ValueArray;

  Edge() { child_AV.fill(0); }

  int E = 0;  // real or virtual count
  float policy_prior_prob = 0;

  // policy_prior_prob + adjustments (from Dirichlet-noise and softmax-temperature)
  float adjusted_base_prob = 0;

  // child_AV is set to with the neural network's AV estimate of this edge's action at the time the
  // parent is evaluated.
  ValueArray child_AV;
};

}  // namespace alpha0
