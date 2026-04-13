#pragma once

// Redirect: search::SearchContext<Spec> → alpha0::SearchContext<Spec::EvalSpec>

#include "alpha0/SearchContext.hpp"
#include "search/concepts/SpecConcept.hpp"

namespace search {

template <search::concepts::Spec Spec>
using SearchContext = alpha0::SearchContext<typename Spec::EvalSpec>;

}  // namespace search
