#pragma once

// Redirect: search::SearchContext<SearchSpec> → alpha0::SearchContext<SearchSpec::EvalSpec>

#include "alpha0/SearchContext.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

namespace search {

template <search::concepts::SearchSpec SearchSpec>
using SearchContext = alpha0::SearchContext<typename SearchSpec::EvalSpec>;

}  // namespace search
