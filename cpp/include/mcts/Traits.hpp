#pragma once

#include "core/concepts/Game.hpp"
#include "mcts/Edge.hpp"
#include "mcts/Node.hpp"
#include "search/concepts/Traits.hpp"

namespace mcts {

template <core::concepts::Game G>
struct Traits {
  using Game = G;
  using Node = mcts::Node<Traits>;
  using Edge = mcts::Edge;

  static_assert(search::concepts::Traits<Traits>);
};

}  // namespace mcts
