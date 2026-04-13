#pragma once

#include "alpha0/NodeStableData.hpp"
#include "alpha0/NodeStats.hpp"
#include "alpha0/concepts/SpecConcept.hpp"
#include "core/Node.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
using Node = core::Node<alpha0::NodeStableData<Spec>, alpha0::NodeStats<Spec>>;

}  // namespace alpha0
