#pragma once

#include "core/concepts/GameConcept.hpp"

namespace core {

/*
 * The DefaultTransposer class provides a default implementation of the Transposer interface
 * required by a class implementing the core::concepts::EvalSpec concept.
 *
 * In DefaultTransposer<Game>, the transpose key for a state is simply the state itself.
 */
template <concepts::Game Game, typename _Key=typename Game::State>
struct DefaultTransposer {
  using State = Game::State;
  using Key = _Key;

  static Key key(const State& state) { return state; }
};

}  // namespace core
