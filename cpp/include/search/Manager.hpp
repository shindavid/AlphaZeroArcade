#pragma once

// Redirect: search::Manager<SearchSpec> → alpha0::Manager<SearchSpec::EvalSpec>
//
// The Manager class has been moved from the search:: namespace to the alpha0:: namespace,
// and re-parameterized from SearchSpec to EvalSpec.

#include "alpha0/Manager.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

namespace search {

template <search::concepts::SearchSpec SearchSpec>
using Manager = alpha0::Manager<typename SearchSpec::EvalSpec>;

}  // namespace search
