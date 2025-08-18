#pragma once

#include "core/BasicTypes.hpp"
#include "search/TypeDefs.hpp"
#include "util/FiniteGroups.hpp"

namespace search {

/*
 * An Edge corresponds to an action that can be taken from this node.
 */
struct Edge {
  node_pool_index_t child_index = -1;
  core::action_t action = -1;
  int E = 0;            // real or virtual count
  float base_prob = 0;  // used for both raw policy prior and chance node probability

  // equal to base_prob, with possible adjustments from Dirichlet-noise and softmax-temperature
  float adjusted_base_prob = 0;

  float child_V_estimate = 0;  // network estimate of child-value for current-player
  group::element_t sym = -1;
  core::context_id_t expanding_context_id = -1;
  expansion_state_t state = kNotExpanded;
};

}  // namespace search
