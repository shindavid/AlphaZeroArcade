#pragma once

#include "alphazero/Algorithms.hpp"
#include "core/concepts/Game.hpp"
#include "search/AlgorithmsFor.hpp"

namespace beta0 {

// forward declaration
template <core::concepts::Game Game>
struct Traits;

}  // namespace beta0

namespace search {

template <core::concepts::Game Game>
struct algorithms_for<beta0::Traits<Game>> {
  // For now, we use the same Algorithms as AlphaZero. Later we will specialize it.
  using type = alpha0::Algorithms<beta0::Traits<Game>>;
};

}  // namespace search
