#pragma once

#include <core/concepts/Game.hpp>
#include <util/BitSet.hpp>
#include <util/FiniteGroups.hpp>

#include <concepts>

namespace core {

/*
 * The DefaultCanonicalizer applies every possible symmetry to a state, sorts the results, and
 * returns the first one.
 *
 * This can be inefficient, but it is guaranteed to work for any game, as long as the game's
 * BaseState class is a comparable type.
 */
template<concepts::Game Game>
class DefaultCanonicalizer {
 public:
  using BaseState = Game::BaseState;

  static group::element_t get(const BaseState& state);
};

}  // namespace core

#include <inline/core/DefaultCanonicalizer.inl>
