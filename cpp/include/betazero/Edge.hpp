#pragma once

#include "alphazero/Edge.hpp"

namespace beta0 {

/*
 * An Edge corresponds to an action that can be taken from this node.
 */
struct Edge : public alpha0::Edge {
  // In alpha0, the edge-count *is* the (unnormalized) posterior policy value.
  //
  // In beta0, we decouple the two concepts, and so track the posterior policy value separately.
  float policy_posterior_prob = 0;

  float child_U_estimate = 0;  // network estimate of child-value-uncertainty for current-player
};

}  // namespace beta0
