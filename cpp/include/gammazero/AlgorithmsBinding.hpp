#pragma once

#include "gammazero/Algorithms.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/AlgorithmsFor.hpp"

namespace gamma0 {

// forward declaration
template <core::concepts::Game Game, core::concepts::EvalSpec EvalSpec>
struct Traits;

}  // namespace gamma0

namespace search {

template <core::concepts::Game Game, core::concepts::EvalSpec EvalSpec>
struct algorithms_for<gamma0::Traits<Game, EvalSpec>> {
  // For now, we use the same Algorithms as AlphaZero. Later we will specialize it.
  using type = gamma0::Algorithms<gamma0::Traits<Game, EvalSpec>>;
};

}  // namespace search
