#pragma once

#include "core/concepts/Game.hpp"

namespace core {

/*
 * The DefaultKeys class provides default implementations of the Keys interface required by a
 * class implementing the core::concepts::EvalSpec concept.
 *
 * In DefaultKeys<Game>, the key used for both transpose_key() and eval_key() is simply the
 * current state.
 */
template <concepts::Game Game>
struct DefaultKeys {
  using State = Game::State;
  using StateHistory = Game::StateHistory;
  using TransposeKey = State;
  using EvalKey = State;

  static TransposeKey transpose_key(const StateHistory& history) { return history.current(); }

  template <util::concepts::RandomAccessIteratorOf<State> Iter>
  static EvalKey eval_key(Iter start, Iter cur) {
    return *cur;
  }
};

}  // namespace core
