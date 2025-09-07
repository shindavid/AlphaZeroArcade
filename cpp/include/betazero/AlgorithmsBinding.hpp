#pragma once

#include "alphazero/Algorithms.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/AlgorithmsFor.hpp"

namespace beta0 {

// forward declaration
template <core::concepts::Game Game, core::concepts::EvalSpec EvalSpec>
struct Traits;

}  // namespace beta0

namespace search {

template <core::concepts::Game Game, core::concepts::EvalSpec EvalSpec>
struct algorithms_for<beta0::Traits<Game, EvalSpec>> {
  // For now, we use the same Algorithms as AlphaZero. Later we will specialize it.
  using type = alpha0::Algorithms<beta0::Traits<Game, EvalSpec>>;
};

}  // namespace search
