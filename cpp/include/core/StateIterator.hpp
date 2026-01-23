#pragma once

#include "core/BasicTypes.hpp"
#include "core/GameStateTree.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
class StateIterator {
 public:
  StateIterator(const GameStateTree<Game>* tree, game_tree_index_t current_index)
      : tree_(tree), index_(current_index) {}

  const Game::State& operator*() const { return tree_->state(index_); }
  StateIterator& operator++();
  StateIterator operator++(int);

  game_tree_node_aux_t get_player_aux() const {
    auto parent_index = tree_->get_parent_index(index_);
    if (parent_index >= 0) {
      return tree_->get_player_aux(parent_index, tree_->get_active_seat(parent_index));
    }
    return 0;
  }

  bool end() const { return index_ < 0; }

 private:
  const GameStateTree<Game>* tree_ = nullptr;
  game_tree_index_t index_ = -1;
};

}  // namespace core

#include "inline/core/StateIterator.inl"
