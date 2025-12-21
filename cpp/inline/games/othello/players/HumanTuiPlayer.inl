#include "games/othello/players/HumanTuiPlayer.hpp"

#include "games/othello/Constants.hpp"

#include <iostream>
#include <string>

namespace othello {

inline core::ActionResponse HumanTuiPlayer::prompt_for_action(const ActionRequest& request) {
  const ActionMask& valid_actions = request.valid_actions;
  bool undo_allowed = request.undo_allowed;

  if (valid_actions[kPass]) {
    std::cout << "Press Enter to pass: ";
    std::cout.flush();
    std::string input;
    std::getline(std::cin, input);
    return kPass;
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
      return core::ActionResponse::undo();
    } else {
      return core::ActionResponse::invalid();
    }
  }

  if (input.size() < 2) {
    return core::ActionResponse::invalid();
  }

  int col = input[0] - 'A';
  if (col < 0 || col >= 8) {
    col = input[0] - 'a';  // accept lower-case
  }
  int row;
  try {
    row = std::stoi(input.substr(1)) - 1;
  } catch (std::invalid_argument& e) {
    return core::ActionResponse::invalid();
  } catch (std::out_of_range& e) {
    return core::ActionResponse::invalid();
  }
  if (col < 0 || col >= kBoardDimension || row < 0 || row >= kBoardDimension) {
    return core::ActionResponse::invalid();
  }

  return row * kBoardDimension + col;
}

}  // namespace othello
