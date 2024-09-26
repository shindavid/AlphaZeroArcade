#include <games/chess/players/HumanTuiPlayer.hpp>

#include <iostream>
#include <string>

namespace chess {

inline core::action_t HumanTuiPlayer::prompt_for_action(const State& state,
                                                        const ActionMask& valid_actions) {
  throw std::runtime_error("Not implemented");
}

}  // namespace chess
