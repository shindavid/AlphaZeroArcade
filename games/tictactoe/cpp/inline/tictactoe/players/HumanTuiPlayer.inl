#include <games/tictactoe/players/HumanTuiPlayer.hpp>

#include <iostream>
#include <string>

namespace tictactoe {

inline HumanTuiPlayer::Action HumanTuiPlayer::prompt_for_action(const GameState& state,
                                                                const ActionMask& valid_actions) {
  Action action;
  action[0] = prompt_for_action_helper(state, valid_actions);
  return action;
}

inline int HumanTuiPlayer::prompt_for_action_helper(const GameState&, const ActionMask&) {
  std::cout << "Enter move [0-8]: ";
  std::cout.flush();
  std::string input;
  std::getline(std::cin, input);
  try {
    return std::stoi(input);
  } catch (std::invalid_argument& e) {
    return -1;
  } catch (std::out_of_range& e) {
    return -1;
  }
}

inline void HumanTuiPlayer::print_state(const GameState& state, bool terminal) {
  state.dump(&last_action_, &this->get_player_names());
}

}  // namespace tictactoe
