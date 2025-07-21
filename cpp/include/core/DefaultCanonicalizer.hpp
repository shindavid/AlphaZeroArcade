#pragma once

#include "core/concepts/Game.hpp"
#include "util/BitSet.hpp"
#include "util/FiniteGroups.hpp"

#include <concepts>

namespace core {

/*
 * The DefaultCanonicalizer applies every possible symmetry to a state, sorts the results, and
 * returns the first one.
 *
 * This can be inefficient, but it is guaranteed to work for any game, as long as the game's
 * State class is a comparable type.
 */
template <concepts::Game Game>
class DefaultCanonicalizer {
 public:
  using State = Game::State;
  static_assert(std::totally_ordered<State>, "State must be totally ordered");

  static group::element_t get(const State& state);
};

}  // namespace core

#include "inline/core/DefaultCanonicalizer.inl"
