#pragma once

#include "core/BasicTypes.hpp"
#include "alpha0/concepts/SpecConcept.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
struct GameLogCompactRecord {
  using InputEncoder = Spec::TensorEncodings::InputEncoder;
  using InputFrame = Spec::InputFrame;
  using Move = Spec::Game::Move;

  InputFrame frame;
  Move move;
  core::seat_index_t active_seat;
};

}  // namespace alpha0
