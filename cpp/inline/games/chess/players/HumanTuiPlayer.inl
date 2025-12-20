#include "games/chess/players/HumanTuiPlayer.hpp"

#include <iostream>
#include <string>

namespace chess {

inline HumanTuiPlayer::ActionResponse HumanTuiPlayer::prompt_for_action(
  const ActionRequest& request) {
  throw std::runtime_error("Not implemented");
}

}  // namespace chess
