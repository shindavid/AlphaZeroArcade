#pragma once

#include "alphazero/NodeStats.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct NodeStats : public alpha0::NodeStats<EvalSpec> {
  float W = 0;  // uncertainty
};

}  // namespace beta0
