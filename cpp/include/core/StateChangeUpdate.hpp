#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

/*
 * StateChangeUpdate is sent to players to notify them of changes in game state. Depending on the
 * type of player, they may use a subset of the information provided here. For example, an alpha0
 * player may only need the action. Webplayer may need the full state to render it on the frontend.
 */
template <concepts::Game Game>
struct StateChangeUpdate {
  using State = Game::State;

  const State& state;
  action_t action;
  game_tree_index_t index;
  game_tree_index_t parent_index;
  seat_index_t seat;
  action_mode_t mode;
};

}  // namespace core
