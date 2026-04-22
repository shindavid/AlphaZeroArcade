#pragma once

#include "beta0/Edge.hpp"
#include "beta0/Node.hpp"
#include "beta0/concepts/SpecConcept.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
struct GraphTraits {
  using Game = Spec::Game;
  using Node = beta0::Node<Spec>;
  using Edge = beta0::Edge<Spec>;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;
  using TransposeKey = typename Spec::Transposer::Key;
};

}  // namespace beta0
