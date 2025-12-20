#include "games/stochastic_nim/players/HumanTuiPlayer.hpp"

#include "util/Asserts.hpp"

#include <iostream>
#include <string>

namespace stochastic_nim {

inline HumanTuiPlayer::ActionResponse HumanTuiPlayer::prompt_for_action(
  const ActionRequest& request) {

  const ActionMask& valid_actions = request.valid_actions;

  int a = -1;
  int b = -1;
  for (int i : valid_actions.on_indices()) {
    if (a == -1) a = i;
    b = i;
  }
  RELEASE_ASSERT(a != -1 && b != -1, "No valid actions");

  a++;
  b++;

  std::cout << "Enter number of stones to take [" << a;
  if (b > a) {
    std::cout << "-" << b;
  }
  std::cout << "]: ";
  std::cout.flush();
  std::string input;
  std::getline(std::cin, input);
  try {
    return ActionResponse::make_move(std::stoi(input) - 1);
  } catch (std::invalid_argument& e) {
    return ActionResponse::invalid();
  } catch (std::out_of_range& e) {
    return ActionResponse::invalid();
  }
}

inline void HumanTuiPlayer::print_state(const State& state, bool terminal) {
  Game::IO::print_state(std::cout, state, last_action_, &this->get_player_names());
}

}  // namespace stochastic_nim
