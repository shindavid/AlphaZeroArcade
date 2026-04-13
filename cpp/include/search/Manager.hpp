#pragma once

// Redirect: search::Manager<Spec> → alpha0::Manager<Spec::EvalSpec>
//
// The Manager class has been moved from the search:: namespace to the alpha0:: namespace,
// and re-parameterized from Spec to EvalSpec.

#include "alpha0/Manager.hpp"
#include "search/concepts/SpecConcept.hpp"

namespace search {

template <search::concepts::Spec Spec>
using Manager = alpha0::Manager<typename Spec::EvalSpec>;

}  // namespace search
