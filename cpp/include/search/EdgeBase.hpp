#pragma once

#include "core/BasicTypes.hpp"
#include "search/TypeDefs.hpp"
#include "util/FiniteGroups.hpp"

namespace search {

template <typename Derived>
struct EdgeBase {
  node_pool_index_t child_index = -1;
  core::action_t action = -1;
  group::element_t sym = -1;
  core::context_id_t expanding_context_id = -1;
  expansion_state_t state = kNotExpanded;
};

}  // namespace search
