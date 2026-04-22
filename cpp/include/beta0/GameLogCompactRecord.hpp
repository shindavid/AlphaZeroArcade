#pragma once

#include "beta0/concepts/SpecConcept.hpp"
#include "core/BasicTypes.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
struct GameLogCompactRecord {
  using InputEncoder = Spec::TensorEncodings::InputEncoder;
  using InputFrame = Spec::InputFrame;
  using Move = Spec::Game::Move;
  using ValueArray = Spec::Game::Types::ValueArray;

  InputFrame frame;
  Move move;
  core::seat_index_t active_seat;

  // W_target: retroactively filled by GameWriteLog::add_terminal() via lambda-discounted Q sums.
  ValueArray W_target;
  bool W_target_valid = false;
};

}  // namespace beta0
