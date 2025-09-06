#pragma once

#include "alphazero/Algorithms.hpp"
#include "core/concepts/Game.hpp"
#include "search/AlgorithmsFor.hpp"

namespace a0 {

// forward declaration
template <core::concepts::Game Game>
struct Traits;

}  // namespace a0

namespace search {

template <core::concepts::Game Game>
struct algorithms_for<a0::Traits<Game>> {
  using type = a0::Algorithms<a0::Traits<Game>>;
};

}  // namespace search
