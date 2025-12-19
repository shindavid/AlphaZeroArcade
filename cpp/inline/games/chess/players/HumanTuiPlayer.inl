#include "games/chess/players/HumanTuiPlayer.hpp"

#include <iostream>
#include <string>

namespace chess {

inline HumanTuiPlayer::ActionResponse HumanTuiPlayer::prompt_for_action(
  const State& state, const ActionMask& valid_actions, bool undo_allowed) {
  throw std::runtime_error("Not implemented");
}

}  // namespace chess
