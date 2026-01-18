#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
struct BacktrackUpdate {
  using State = Game::State;
  using History = std::vector<const State*>;

  const History& history;
  action_t action;
  game_tree_index_t index;
  action_mode_t mode;
};

}  // namespace core
