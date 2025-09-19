#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace alpha0 {

template <core::concepts::Game Game>
struct GameLogCompactRecord {
  using State = Game::State;

  State position;
  core::seat_index_t active_seat;
  core::action_mode_t action_mode;
  core::action_t action;
};

}  // namespace alpha0
