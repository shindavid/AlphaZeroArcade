#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
struct GameLogCompactRecord {
  using InputTensorizor = EvalSpec::InputTensorizor;
  using InputFrame = EvalSpec::InputFrame;

  InputFrame frame;
  core::seat_index_t active_seat;
  core::action_mode_t action_mode;
  core::action_t action;
};

}  // namespace alpha0
