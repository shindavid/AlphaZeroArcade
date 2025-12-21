#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
struct StateChangeUpdate {
  using State = Game::State;

  seat_index_t seat;
  const State& state;
  action_t action;
  game_tree_index_t game_tree_index;
  action_mode_t action_mode;
};

}  // namespace core
