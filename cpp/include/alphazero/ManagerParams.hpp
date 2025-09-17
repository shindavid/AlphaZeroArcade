#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "search/ManagerParamsBase.hpp"


namespace alpha0 {

// For now, most of the code lives in ManagerParamsBase, because beta0 is currently just a copy of
// alpha0. As we specialize beta0 more, we should move more code from ManagerParamsBase to
// alpha0::ManagerParams.
template <core::concepts::EvalSpec EvalSpec>
struct ManagerParams : public search::ManagerParamsBase<EvalSpec> {};

}  // namespace alpha0
