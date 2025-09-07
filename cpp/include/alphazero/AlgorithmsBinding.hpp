#pragma once

#include "alphazero/Algorithms.hpp"
#include "core/concepts/Game.hpp"
#include "search/AlgorithmsFor.hpp"

namespace alpha0 {

// forward declaration
template <core::concepts::Game Game>
struct Traits;

}  // namespace alpha0

namespace search {

template <core::concepts::Game Game>
struct algorithms_for<alpha0::Traits<Game>> {
  using type = alpha0::Algorithms<alpha0::Traits<Game>>;
};

}  // namespace search
