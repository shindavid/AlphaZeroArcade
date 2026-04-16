#pragma once

#include "core/BasicTypes.hpp"
#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

/*
 * StateChangeUpdate is sent to players to notify them of changes in game state. Depending on the
 * type of player, they may use a subset of the information provided here. For example, an alpha0
 * player may only need the action. Webplayer may need the full state to render it on the frontend.
 *
 * The info_set_it() accessor returns an InfoSetIterator, which yields the InfoSet visible to the
 * target player at each node in the history.
 */
template <concepts::Game Game>
struct StateChangeUpdate {
  using InfoSet = Game::InfoSet;
  using Move = Game::Move;
  using InfoSetIterator = core::InfoSetIterator<Game>;

  StateChangeUpdate(InfoSetIterator it, const Move* m, game_tree_index_t i, game_tree_index_t pi,
                    step_t st, seat_index_t se, bool j = false)
      : info_set_it_(it),
        index_(i),
        parent_index_(pi),
        step_(st),
        seat_(se),
        jump_(j),
        move_is_valid_(m != nullptr) {
    if (m) move_ = *m;
  }

  StateChangeUpdate(InfoSetIterator it, const Move* m, step_t st, seat_index_t se)
      : info_set_it_(it),
        index_(-1),
        parent_index_(-1),
        step_(st),
        seat_(se),
        jump_(false),
        move_is_valid_(m != nullptr) {
    if (m) move_ = *m;
  }

  InfoSetIterator info_set_it() const { return info_set_it_; }

  const Move* move() const { return move_is_valid_ ? &move_ : nullptr; }
  game_tree_index_t index() const { return index_; }
  game_tree_index_t parent_index() const { return parent_index_; }
  step_t step() const { return step_; }
  seat_index_t seat() const { return seat_; }
  bool is_jump() const { return jump_; }

 private:
  InfoSetIterator info_set_it_;
  Move move_;
  game_tree_index_t index_;
  game_tree_index_t parent_index_;
  step_t step_;
  seat_index_t seat_;
  bool jump_;
  bool move_is_valid_;
};

}  // namespace core
