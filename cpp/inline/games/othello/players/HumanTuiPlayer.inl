#include "games/othello/players/HumanTuiPlayer.hpp"

#include "games/othello/Constants.hpp"

#include <iostream>
#include <string>

namespace othello {

inline core::action_t HumanTuiPlayer::prompt_for_action(const State& state,
                                                        const ActionMask& valid_actions) {
  if (valid_actions[kPass]) {
    std::cout << "Press Enter to pass: ";
    std::cout.flush();
    std::string input;
    std::getline(std::cin, input);
    return kPass;
  }
  std::cout << "Enter move [A1-H8] or U to undo: ";
  std::cout.flush();
  std::string input;
  std::getline(std::cin, input);

  if (input == "U" || input == "u") {
    return generic::HumanTuiPlayer<Game>::kUndoAction;
  }

  if (input.size() < 2) {
    return core::kNullAction;
  }

  int col = input[0] - 'A';
  if (col < 0 || col >= 8) {
    col = input[0] - 'a';  // accept lower-case
  }
  int row;
  try {
    row = std::stoi(input.substr(1)) - 1;
  } catch (std::invalid_argument& e) {
    return core::kNullAction;
  } catch (std::out_of_range& e) {
    return core::kNullAction;
  }
  if (col < 0 || col >= kBoardDimension || row < 0 || row >= kBoardDimension) {
    return core::kNullAction;
  }

  return row * kBoardDimension + col;
}

}  // namespace othello
