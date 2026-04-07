#pragma once

#include "alpha0/Algorithms.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/AlgorithmsFor.hpp"

namespace alpha0 {

// forward declaration
template <core::concepts::Game Game, core::concepts::EvalSpec EvalSpec>
struct SearchSpec;

}  // namespace alpha0

namespace search {

template <core::concepts::Game Game, core::concepts::EvalSpec EvalSpec>
struct algorithms_for<alpha0::SearchSpec<Game, EvalSpec>> {
  using type = alpha0::Algorithms<alpha0::SearchSpec<Game, EvalSpec>>;
};

}  // namespace search
