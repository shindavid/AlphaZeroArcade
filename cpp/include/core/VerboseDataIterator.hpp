#pragma once

#include "core/GameStateTree.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/VerboseDataBase.hpp"

namespace core {

template <concepts::Game Game>
class VerboseDataIterator {
 public:
  using VerboseData_sptr = std::shared_ptr<const generic::VerboseDataBase>;

  VerboseDataIterator() = default;
  VerboseDataIterator(GameStateTree<Game>* tree, game_tree_index_t current_index)
      : tree_(tree), index_(current_index) {}

  VerboseData_sptr& operator*() const { return tree_->verbose_data(index_); }
  VerboseData_sptr most_recent_data() const;
  game_tree_index_t index() const { return index_; }

 private:
  GameStateTree<Game>* tree_ = nullptr;
  game_tree_index_t index_ = -1;
};

}  // namespace core

#include "inline/core/VerboseDataIterator.inl"
