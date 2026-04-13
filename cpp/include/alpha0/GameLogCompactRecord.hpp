#pragma once

#include "core/BasicTypes.hpp"
#include "alpha0/concepts/SpecConcept.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec EvalSpec>
struct GameLogCompactRecord {
  using InputEncoder = EvalSpec::TensorEncodings::InputEncoder;
  using InputFrame = EvalSpec::InputFrame;
  using Move = EvalSpec::Game::Move;

  InputFrame frame;
  Move move;
  core::seat_index_t active_seat;
};

}  // namespace alpha0
