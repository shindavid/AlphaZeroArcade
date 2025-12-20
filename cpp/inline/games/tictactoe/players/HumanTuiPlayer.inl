#include "games/tictactoe/players/HumanTuiPlayer.hpp"

#include <iostream>
#include <string>

namespace tictactoe {

inline HumanTuiPlayer::ActionResponse HumanTuiPlayer::prompt_for_action(
  const ActionRequest& request) {

  std::cout << "Enter move [0-8]: ";
  std::cout.flush();
  std::string input;
  std::getline(std::cin, input);
  try {
    return std::stoi(input);
  } catch (std::invalid_argument& e) {
    return ActionResponse::invalid();
  } catch (std::out_of_range& e) {
    return ActionResponse::invalid();
  }
}

inline void HumanTuiPlayer::print_state(const State& state, bool terminal) {
  Game::IO::print_state(std::cout, state, last_action_, &this->get_player_names());
}

}  // namespace tictactoe
