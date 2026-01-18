#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
struct BacktrackUpdate {
  using State = Game::State;
  using ReverseHistory = std::vector<const State*>;  // ReverseHistory[0] is the most recent state.

  const ReverseHistory& reverse_history;
  action_t action;
  game_tree_index_t index;
  step_t step;
  action_mode_t mode;

};

}  // namespace core
