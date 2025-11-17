#pragma once

#include "search/ManagerWithSymmetryTranspositions.hpp"
#include "search/SimpleManager.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <type_traits>

namespace search {

template <search::concepts::Traits Traits>
using Manager =
  std::conditional_t<Traits::EvalSpec::MctsConfiguration::kSupportSymmetryTranspositions,
                     ManagerWithSymmetryTranspositions<Traits>,
                     SimpleManager<Traits>>;

}  // namespace search
