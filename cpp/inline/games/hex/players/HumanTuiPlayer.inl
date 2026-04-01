#include "games/hex/players/HumanTuiPlayer.hpp"

#include "games/hex/Constants.hpp"

#include <format>
#include <iostream>
#include <string>

namespace hex {

inline core::ActionResponse HumanTuiPlayer::prompt_for_action(const ActionRequest& request) {
  const State& state = request.state;

  constexpr int B = Constants::kBoardDim;
  bool can_swap = state.core.cur_player == Constants::kSecondPlayer && !state.core.post_swap_phase;

  std::string prompt = std::format("Enter move [A1-K11{}]: ", can_swap ? " (or S to swap)" : "");
  std::cout << prompt;
  std::cout.flush();
  std::string input;
  std::getline(std::cin, input);

  if (input.empty()) {
    return core::ActionResponse::invalid();  // no input
  }
  if (input == "S" || input == "s") {
    if (can_swap) {
      return kSwap;  // swap action
    } else {
      return core::ActionResponse::invalid();  // invalid swap
    }
  }

  int col = input[0] - 'A';
  if (col < 0 || col >= B) {
    col = input[0] - 'a';  // accept lower-case
  }
  if (col < 0 || col >= B) {
    return core::ActionResponse::invalid();  // invalid column
  }

  int row;
  try {
    row = std::stoi(input.substr(1)) - 1;
  } catch (std::invalid_argument& e) {
    return core::ActionResponse::invalid();
  } catch (std::out_of_range& e) {
    return core::ActionResponse::invalid();
  }
  if (row < 0 || row >= B) {
    return core::ActionResponse::invalid();
  }

  return row * B + col;
}

}  // namespace hex
