#include "games/hex/players/HumanTuiPlayer.hpp"

#include "games/hex/Constants.hpp"

#include <format>
#include <iostream>
#include <string>

namespace hex {

inline HumanTuiPlayer::ActionResponse HumanTuiPlayer::prompt_for_action(
  const ActionRequest& request) {
  const State& state = request.state;

  constexpr int B = Constants::kBoardDim;
  bool can_swap = state.core.cur_player == Constants::kSecondPlayer && !state.core.post_swap_phase;

  std::string prompt = std::format("Enter move [A1-K11{}]: ", can_swap ? " (or S to swap)" : "");
  std::cout << prompt;
  std::cout.flush();
  std::string input;
  std::getline(std::cin, input);

  if (input.empty()) {
    return ActionResponse::invalid();  // no input
  }
  if (input == "S" || input == "s") {
    if (can_swap) {
      return ActionResponse::make_move(kSwap);  // swap action
    } else {
      return ActionResponse::invalid();  // invalid swap
    }
  }

  int col = input[0] - 'A';
  if (col < 0 || col >= B) {
    col = input[0] - 'a';  // accept lower-case
  }
  if (col < 0 || col >= B) {
    return ActionResponse::invalid();  // invalid column
  }

  int row;
  try {
    row = std::stoi(input.substr(1)) - 1;
  } catch (std::invalid_argument& e) {
    return ActionResponse::invalid();
  } catch (std::out_of_range& e) {
    return ActionResponse::invalid();
  }
  if (row < 0 || row >= B) {
    return ActionResponse::invalid();
  }

  return ActionResponse::make_move(row * B + col);
}

}  // namespace hex
