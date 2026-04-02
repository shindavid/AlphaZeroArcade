#include "games/stochastic_nim/players/HumanTuiPlayer.hpp"

#include "util/Asserts.hpp"

#include <iostream>
#include <string>

namespace stochastic_nim {

inline HumanTuiPlayer::ActionResponse HumanTuiPlayer::prompt_for_action(
  const ActionRequest& request) {
  const MoveList& valid_moves = request.valid_moves;

  int a = -1;
  int b = -1;
  for (Move move : valid_moves) {
    if (a == -1) a = move.index();
    b = move.index();
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
    return Move(std::stoi(input) - 1, stochastic_nim::kPlayerPhase);
  } catch (std::invalid_argument& e) {
    return ActionResponse::invalid();
  } catch (std::out_of_range& e) {
    return ActionResponse::invalid();
  }
}

inline void HumanTuiPlayer::print_state(const State& state, bool terminal) {
  Game::IO::print_state(std::cout, state, &last_move_, &this->get_player_names());
}

}  // namespace stochastic_nim
