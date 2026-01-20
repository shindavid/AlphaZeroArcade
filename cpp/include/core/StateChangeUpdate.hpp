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

  // TODO: move to private and make getters

  StateChangeUpdate(StateIterator it, action_t a, game_tree_index_t i, game_tree_index_t pi,
                    step_t st, seat_index_t se, action_mode_t m, bool j = false)
      : state_it_(it),
        action_(a),
        index_(i),
        parent_index_(pi),
        step_(st),
        seat_(se),
        mode_(m),
        jump_(j) {}

  StateChangeUpdate(StateIterator it, action_t a, step_t st, seat_index_t se)
      : state_it_(it),
        action_(a),
        index_(-1),
        parent_index_(-1),
        step_(st),
        seat_(se),
        mode_(-1),
        jump_(false) {}

  StateIterator state_it() const { return state_it_; }
  action_t action() const { return action_; }
  game_tree_index_t index() const { return index_; }
  game_tree_index_t parent_index() const { return parent_index_; }
  step_t step() const { return step_; }
  seat_index_t seat() const { return seat_; }
  action_mode_t mode() const { return mode_; }
  bool is_jump() const { return jump_; }

 private:
  StateIterator state_it_;
  action_t action_;
  game_tree_index_t index_;
  game_tree_index_t parent_index_;
  step_t step_;
  seat_index_t seat_;
  action_mode_t mode_;
  bool jump_;
};

}  // namespace core
