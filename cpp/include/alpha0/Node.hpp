#pragma once

#include "alpha0/NodeStableData.hpp"
#include "alpha0/NodeStats.hpp"
#include "alpha0/concepts/EvalSpecConcept.hpp"
#include "core/Node.hpp"

namespace alpha0 {

template <alpha0::concepts::EvalSpec EvalSpec>
using Node = core::Node<alpha0::NodeStableData<EvalSpec>, alpha0::NodeStats<EvalSpec>>;

}  // namespace alpha0
