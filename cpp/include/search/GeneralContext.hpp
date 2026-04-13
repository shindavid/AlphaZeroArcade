#pragma once

// Redirect: search::GeneralContext<Spec> → alpha0::GeneralContext<Spec::EvalSpec>

#include "alpha0/GeneralContext.hpp"
#include "search/concepts/SpecConcept.hpp"

namespace search {

template <search::concepts::Spec Spec>
using GeneralContext = alpha0::GeneralContext<typename Spec::EvalSpec>;

}  // namespace search
