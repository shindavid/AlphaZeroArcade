#pragma once

#include "core/BasicTypes.hpp"
#include "alpha0/concepts/SpecConcept.hpp"

#include <cstdint>

namespace search {

template <::alpha0::concepts::Spec Spec>
struct EdgeBase {
  using Move = Spec::Game::Move;

  enum expansion_state_t : int8_t {
    kNotExpanded,
    kMidExpansion,
    kPreExpanded,  // used when evaluating all children when computing AV-targets
    kExpanded
  };

  core::node_pool_index_t child_index = -1;
  Move move;
  float chance_prob = 0;  // only valid for chance nodes
  core::context_id_t expanding_context_id = -1;
  expansion_state_t state = kNotExpanded;
  bool was_pre_expanded = false;
};

}  // namespace search
