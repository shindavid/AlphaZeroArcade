#pragma once

// Redirect: search::GeneralContext<SearchSpec> → alpha0::GeneralContext<SearchSpec::EvalSpec>

#include "alpha0/GeneralContext.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

namespace search {

template <search::concepts::SearchSpec SearchSpec>
using GeneralContext = alpha0::GeneralContext<typename SearchSpec::EvalSpec>;

}  // namespace search
