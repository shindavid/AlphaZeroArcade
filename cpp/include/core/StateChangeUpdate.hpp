#pragma once

#include "core/BasicTypes.hpp"
#include "core/StateIterator.hpp"
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
  using StateIterator = core::StateIterator<Game>;

  StateIterator state_it;
  action_t action;
  game_tree_index_t index;
  game_tree_index_t parent_index;
  step_t step;
  seat_index_t seat;
  action_mode_t mode;
  bool jump;

  StateChangeUpdate(StateIterator it, action_t a, game_tree_index_t i, game_tree_index_t pi,
                    step_t st, seat_index_t se, action_mode_t m, bool j = false)
      : state_it(it), action(a), index(i), parent_index(pi), step(st), seat(se), mode(m), jump(j) {}

  StateChangeUpdate(StateIterator it, action_t a, step_t st, seat_index_t se)
      : state_it(it), action(a), index(-1), parent_index(-1), step(st), seat(se), mode(-1) {}
};

}  // namespace core
