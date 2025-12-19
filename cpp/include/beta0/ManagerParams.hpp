#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "search/ManagerParamsBase.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct ManagerParams : public search::ManagerParamsBase<EvalSpec> {};

}  // namespace beta0
