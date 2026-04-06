#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
struct GameLogCompactRecord {
  using InputEncoder = EvalSpec::TensorEncodings::InputEncoder;
  using InputFrame = EvalSpec::InputFrame;
  using Move = EvalSpec::Game::Move;

  InputFrame frame;
  Move move;
  core::seat_index_t active_seat;
};

}  // namespace alpha0
