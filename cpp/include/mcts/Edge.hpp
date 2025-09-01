#pragma once

#include "search/EdgeBase.hpp"

namespace mcts {

/*
 * An Edge corresponds to an action that can be taken from this node.
 */
struct Edge : public search::EdgeBase {
  int E = 0;  // real or virtual count
  float policy_prior_prob = 0;

  // policy_prior_prob + adjustments (from Dirichlet-noise and softmax-temperature)
  float adjusted_base_prob = 0;

  float child_V_estimate = 0;  // network estimate of child-value for current-player
};

}  // namespace mcts
