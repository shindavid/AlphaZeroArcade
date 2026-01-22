#pragma once

#include "core/BasicTypes.hpp"
#include "core/GameStateTree.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/VerboseDataBase.hpp"

namespace core {

template <concepts::Game Game>
class StateIterator {
 public:
  StateIterator(const GameStateTree<Game>* tree, game_tree_index_t current_index)
      : tree_(tree), index_(current_index) {}

  const Game::State& operator*() const { return tree_->state(index_); }
  StateIterator& operator++();
  StateIterator operator++(int);

  bool end() const { return index_ < 0; }

 private:
  const GameStateTree<Game>* tree_ = nullptr;
  game_tree_index_t index_ = -1;
};

template <concepts::Game Game>
class VerboseDataIterator {
 public:
  using VerboseDataPtr = generic::VerboseDataBase*;
  VerboseDataIterator() = default;
  VerboseDataIterator(GameStateTree<Game>* tree, game_tree_index_t current_index)
      : tree_(tree), index_(current_index) {}

  VerboseDataPtr& operator*() const { return tree_->verbose_data(index_); }
  VerboseDataPtr most_recent_data() const;
  VerboseDataIterator& operator++();
  VerboseDataIterator operator++(int);

  bool end() const { return index_ < 0; }
  game_tree_index_t index() const { return index_; }

 private:
  GameStateTree<Game>* tree_ = nullptr;
  game_tree_index_t index_ = -1;
};

}  // namespace core

#include "inline/core/StateIterator.inl"
