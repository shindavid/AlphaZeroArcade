#pragma once

#include "core/BasicTypes.hpp"
#include "core/GameStateTree.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

/*
 * InfoSetIterator traverses the game history from the perspective of a specific player,
 * yielding InfoSet references at each node. For perfect-info games (InfoSet == State),
 * this is identical to the old StateIterator.
 */
template <concepts::Game Game>
class InfoSetIterator {
 public:
  using Move = Game::Move;
  using InfoSet = Game::InfoSet;

  InfoSetIterator(const GameStateTree<Game>* tree, game_tree_index_t current_index,
                  seat_index_t seat)
      : tree_(tree), index_(current_index), seat_(seat) {}

  struct NodeData {
    const InfoSet& info_set;
    game_tree_node_aux_t aux;
    const Move* move_from_parent;
    step_t step;
    const NodeData* operator->() { return this; }
  };

  NodeData operator*() const {
    return {tree_->info_set(index_, seat_), get_player_aux(), tree_->get_move_from_parent(index_),
            tree_->get_step(index_)};
  }
  NodeData operator->() const { return **this; }

  InfoSetIterator& operator++();
  InfoSetIterator operator++(int);
  bool end() const { return index_ < 0; }

  // Expose index for callers that need it (e.g., aux lookup)
  game_tree_index_t index() const { return index_; }
  seat_index_t seat() const { return seat_; }

 private:
  game_tree_node_aux_t get_player_aux() const;

  const GameStateTree<Game>* tree_ = nullptr;
  game_tree_index_t index_ = -1;
  seat_index_t seat_ = -1;
};

}  // namespace core

#include "inline/core/StateIterator.inl"
