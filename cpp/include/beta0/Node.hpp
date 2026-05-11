#pragma once

#include "beta0/NodeStableData.hpp"
#include "beta0/NodeStats.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "core/Node.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
using Node = core::Node<beta0::NodeStableData<Spec>, beta0::NodeStats<Spec>>;

}  // namespace beta0
