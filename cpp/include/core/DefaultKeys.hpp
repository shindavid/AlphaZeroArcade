#pragma once

#include "core/InputTensorizor.hpp"
#include "core/concepts/GameConcept.hpp"

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
  using TransposeKey = State;
  using EvalKey = State;
  using InputTensorizor = core::InputTensorizor<Game>;

  static TransposeKey transpose_key(const State& state) { return state; }

  static EvalKey eval_key(InputTensorizor* input_tensorizor) {
    return input_tensorizor->current_state();
  }

};

}  // namespace core
