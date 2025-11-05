#pragma once

#include "alphazero/Edge.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace gamma0 {

/*
 * An Edge corresponds to an action that can be taken from this node.
 */
template <core::concepts::EvalSpec EvalSpec>
struct Edge : public alpha0::Edge<EvalSpec> {
  using ValueArray = EvalSpec::Game::Types::ValueArray;
  using Base = alpha0::Edge<EvalSpec>;

  Edge() : Base() {
    child_Qgamma_snapshot.fill(0);
    child_W_snapshot.fill(0);
  }

  // In alpha0, the edge-count *is* the (unnormalized) posterior policy value.
  //
  // In gamma0, we decouple the two concepts, and so track the posterior policy value separately.
  float policy_posterior_prob = 0;

  // We track snapshots of the child node's W/Qgamma to support short-circuiting.
  //
  // Note that in alpha0, the only child-stat that needed such tracking was RN, and Edge::E served
  // that purpose.
  //
  // The word "snapshot" is used to emphasize that these fields may be out-of-date, due to MCTS's
  // move transposition mechanism.
  ValueArray child_Qgamma_snapshot;
  ValueArray child_W_snapshot;
};

}  // namespace gamma0
