#include <games/blokus/players/HumanTuiPlayer.hpp>

#include <games/blokus/Constants.hpp>

#include <iostream>
#include <string>

namespace blokus {

inline core::action_t HumanTuiPlayer::prompt_for_action(const FullState& state,
                                                        const ActionMask& valid_actions) {
  throw std::runtime_error("Not implemented");
}

}  // namespace blokus
