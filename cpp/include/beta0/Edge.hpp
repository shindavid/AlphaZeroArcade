#pragma once

#include "beta0/concepts/SpecConcept.hpp"
#include "search/EdgeBase.hpp"

namespace beta0 {

/*
 * An Edge corresponds to an action that can be taken from this node.
 *
 * Extends alpha0::Edge with child_AU (action-value uncertainty) per-player array.
 */
template <beta0::concepts::Spec Spec>
struct Edge : public search::EdgeBase<typename Spec::Game> {
  using ValueArray = Spec::Game::Types::ValueArray;

  Edge() {
    child_AV.fill(0);
    child_AU.fill(0);
  }

  int E = 0;  // real or virtual count
  float policy_prior_prob = 0;

  // policy_prior_prob + adjustments (from Dirichlet-noise and softmax-temperature)
  float adjusted_base_prob = 0;

  // child_AV is set with the neural network's AV estimate of this edge's action at the time the
  // parent is evaluated.
  ValueArray child_AV;

  // child_AU is set with the neural network's AU (action-value uncertainty) estimate at the time
  // the parent is evaluated.
  ValueArray child_AU;
};

}  // namespace beta0
