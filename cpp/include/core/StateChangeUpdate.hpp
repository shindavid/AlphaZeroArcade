#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
struct StateChangeUpdate {
  using State = Game::State;

  const State& state;
  action_t action;
  game_tree_index_t game_tree_index;
  seat_index_t seat;
  bool is_chance;
};

}  // namespace core
