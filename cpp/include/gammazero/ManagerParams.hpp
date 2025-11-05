#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "search/ManagerParamsBase.hpp"

namespace gamma0 {

// For now, most of the code lives in ManagerParamsBase, because gamma0 is currently just a copy of
// alpha0. As we specialize gamma0 more, we should move more code from ManagerParamsBase to
// gamma0::ManagerParams.
template <core::concepts::EvalSpec EvalSpec>
struct ManagerParams : public search::ManagerParamsBase<EvalSpec> {};

}  // namespace gamma0
