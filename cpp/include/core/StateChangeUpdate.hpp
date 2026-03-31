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
  using Move = Game::Move;
  using StateIterator = core::StateIterator<Game>;

  StateChangeUpdate(StateIterator it, Move m, game_tree_index_t i, game_tree_index_t pi, step_t st,
                    seat_index_t se, game_phase_t gp, bool j = false)
      : state_it_(it),
        move_(m),
        index_(i),
        parent_index_(pi),
        step_(st),
        seat_(se),
        game_phase_(gp),
        jump_(j) {}

  StateChangeUpdate(StateIterator it, Move m, step_t st, seat_index_t se)
      : state_it_(it),
        move_(m),
        index_(-1),
        parent_index_(-1),
        step_(st),
        seat_(se),
        game_phase_(-1),
        jump_(false) {}

  StateIterator state_it() const { return state_it_; }
  Move move() const { return move_; }
  game_tree_index_t index() const { return index_; }
  game_tree_index_t parent_index() const { return parent_index_; }
  step_t step() const { return step_; }
  seat_index_t seat() const { return seat_; }
  game_phase_t game_phase() const { return game_phase_; }
  bool is_jump() const { return jump_; }

 private:
  StateIterator state_it_;
  Move move_;
  game_tree_index_t index_;
  game_tree_index_t parent_index_;
  step_t step_;
  seat_index_t seat_;
  game_phase_t game_phase_;
  bool jump_;
};

}  // namespace core
