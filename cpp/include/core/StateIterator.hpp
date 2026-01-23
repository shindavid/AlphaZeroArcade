#pragma once

#include "core/BasicTypes.hpp"
#include "core/GameStateTree.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
class StateIterator {
 public:
  using State = Game::State;

  StateIterator(const GameStateTree<Game>* tree, game_tree_index_t current_index)
      : tree_(tree), index_(current_index) {}

  struct NodeData {
    const State& state;
    game_tree_node_aux_t aux;
    const NodeData* operator->() { return this; }
  };

  NodeData operator*() const { return {tree_->state(index_), get_player_aux()}; }
  NodeData operator->() const { return **this; }

  StateIterator& operator++();
  StateIterator operator++(int);
  bool end() const { return index_ < 0; }

 private:
  game_tree_node_aux_t get_player_aux() const;

  const GameStateTree<Game>* tree_ = nullptr;
  game_tree_index_t index_ = -1;
};

}  // namespace core

#include "inline/core/StateIterator.inl"
