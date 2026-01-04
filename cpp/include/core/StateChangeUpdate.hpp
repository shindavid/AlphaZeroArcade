#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

/*
 * StateChangeUpdate is sent to players to notify them of changes in game state. Depending on the
 * type of player, they may use a subset of the information provided here. For example, an alpha0
 * player may only need the action. Webplayer may need the full state to render it on the frontend.
 * NOTE:
 * 1.`state` is the new state AFTER the `action` has been applied.
 * 2.'game_tree_index` is the index of the new state in the game tree.
 * 3.`seat` is the seat index of the player who made the action and NOT the active player for
`state`.
 * 4. similarly, `action_mode` is the action mode used by the player who made the action.
 */

template <concepts::Game Game>
struct StateChangeUpdate {
  using State = Game::State;

  const State& state;
  action_t action;
  game_tree_index_t game_tree_index;
  seat_index_t seat;
  action_mode_t action_mode;
};

}  // namespace core
