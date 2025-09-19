#pragma once

#include "alphazero/Edge.hpp"

namespace beta0 {

/*
 * An Edge corresponds to an action that can be taken from this node.
 */
struct Edge : public alpha0::Edge {
  float child_U_estimate = 0;  // network estimate of child-value-uncertainty for current-player
};

}  // namespace beta0
