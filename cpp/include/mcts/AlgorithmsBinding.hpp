#pragma once

#include "core/concepts/Game.hpp"
#include "mcts/Algorithms.hpp"
#include "search/AlgorithmsFor.hpp"

namespace mcts {

// forward declaration
template <core::concepts::Game Game>
struct Traits;

}  // namespace mcts

namespace search {

template <core::concepts::Game Game>
struct algorithms_for<mcts::Traits<Game>> {
  using type = mcts::Algorithms<mcts::Traits<Game>>;
};

}  // namespace search
