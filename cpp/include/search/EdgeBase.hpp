#pragma once

#include "core/BasicTypes.hpp"
#include "util/FiniteGroups.hpp"

#include <cstdint>

namespace search {

struct EdgeBase {
  enum expansion_state_t : int8_t {
    kNotExpanded,
    kMidExpansion,
    kPreExpanded,  // used when evaluating all children when computing AV-targets
    kExpanded
  };

  core::node_pool_index_t child_index = -1;
  core::action_t action = -1;
  float chance_prob = 0;  // only valid for chance nodes
  core::context_id_t expanding_context_id = -1;
  expansion_state_t state = kNotExpanded;
};

}  // namespace search
