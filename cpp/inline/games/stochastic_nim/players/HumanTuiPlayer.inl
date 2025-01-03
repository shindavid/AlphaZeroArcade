#include <games/stochastic_nim/players/HumanTuiPlayer.hpp>

#include <iostream>
#include <string>

namespace stochastic_nim {

inline core::action_t HumanTuiPlayer::prompt_for_action(const State& state,
                                                        const ActionMask& valid_actions) {
  std::cout << "Enter move [0-2]: ";
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

inline void HumanTuiPlayer::print_state(const State& state, bool terminal) {
  Game::IO::print_state(std::cout, state, last_action_, &this->get_player_names());
}

}  // namespace stochastic_nim
