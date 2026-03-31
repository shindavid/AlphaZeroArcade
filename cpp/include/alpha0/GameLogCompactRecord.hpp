#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
struct GameLogCompactRecord {
  using InputTensorizor = EvalSpec::InputTensorizor;
  using InputFrame = EvalSpec::InputFrame;
  using Move = EvalSpec::Game::Move;

  InputFrame frame;
  Move move;
  core::game_phase_t game_phase;
  core::seat_index_t active_seat;
};

}  // namespace alpha0
