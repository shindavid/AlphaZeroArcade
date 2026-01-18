#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
struct BacktrackUpdate {
  using State = Game::State;

  // History is in reverse chronological order: History[0] is the latest state
  using History = std::vector<const State*>;

  const History& history;
  action_t action;
  game_tree_index_t index;
  step_t step;
  action_mode_t mode;

};

}  // namespace core
