#include "games/othello/players/HumanTuiPlayer.hpp"

#include "games/othello/Constants.hpp"

#include <iostream>
#include <string>

namespace othello {

inline HumanTuiPlayer::ActionResponse HumanTuiPlayer::prompt_for_action(
  const State& state, const ActionMask& valid_actions, bool undo_allowed) {
  if (valid_actions[kPass]) {
    std::cout << "Press Enter to pass: ";
    std::cout.flush();
    std::string input;
    std::getline(std::cin, input);
    return ActionResponse(kPass);
  }

  if (undo_allowed) {
    std::cout << "Enter move [A1-H8] or U to undo: ";
  } else {
    std::cout << "Enter move [A1-H8]: ";
  }
  std::cout.flush();
  std::string input;
  std::getline(std::cin, input);

  if (input == "U" || input == "u") {
    if (undo_allowed) {
      return ActionResponse::undo();
    } else {
      return ActionResponse::invalid();
    }
  }

  if (input.size() < 2) {
    return ActionResponse::invalid();
  }

  int col = input[0] - 'A';
  if (col < 0 || col >= 8) {
    col = input[0] - 'a';  // accept lower-case
  }
  int row;
  try {
    row = std::stoi(input.substr(1)) - 1;
  } catch (std::invalid_argument& e) {
    return ActionResponse::invalid();
  } catch (std::out_of_range& e) {
    return ActionResponse::invalid();
  }
  if (col < 0 || col >= kBoardDimension || row < 0 || row >= kBoardDimension) {
    return ActionResponse::invalid();
  }

  return row * kBoardDimension + col;
}

}  // namespace othello
