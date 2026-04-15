#pragma once

#include "alpha0/Edge.hpp"
#include "alpha0/Node.hpp"
#include "alpha0/concepts/SpecConcept.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
struct GraphTraits {
  using Game = Spec::Game;
  using Node = alpha0::Node<Spec>;
  using Edge = alpha0::Edge<Spec>;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;
  using TransposeKey = typename Spec::Transposer::Key;
};

}  // namespace alpha0
