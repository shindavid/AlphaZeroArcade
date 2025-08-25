#pragma once

#include "core/BasicTypes.hpp"
#include "search/TypeDefs.hpp"
#include "util/FiniteGroups.hpp"

namespace search {

struct EdgeBase {
  node_pool_index_t child_index = -1;
  core::action_t action = -1;
  float chance_prob = 0;  // only valid for chance nodes
  group::element_t sym = -1;
  core::context_id_t expanding_context_id = -1;
  expansion_state_t state = kNotExpanded;
};

}  // namespace search
