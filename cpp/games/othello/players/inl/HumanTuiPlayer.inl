#include <games/othello/players/HumanTuiPlayer.hpp>

#include <games/othello/Constants.hpp>

#include <iostream>
#include <string>

namespace othello {

inline HumanTuiPlayer::Action HumanTuiPlayer::prompt_for_action(
  const GameState& state, const ActionMask& valid_actions)
{
  Action action;
  action[0] = prompt_for_action_helper(state, valid_actions);
  return action;
}

inline int HumanTuiPlayer::prompt_for_action_helper(const GameState&, const ActionMask& valid_actions) {
  if (valid_actions[kPass]) {
    std::cout << "Press Enter to pass: ";
    std::cout.flush();
    std::string input;
    std::getline(std::cin, input);
    return kPass;
  }
  std::cout << "Enter move [A1-H8]: ";
  std::cout.flush();
  std::string input;
  std::getline(std::cin, input);

  if (input.size() < 2) {
    return -1;
  }
  int col = input[0] - 'A';
  if (col < 0 || col >= 8) {
    col = input[0] - 'a';  // accept lower-case
  }
  int row;
  try {
    row = std::stoi(input.substr(1)) - 1;
  } catch (std::invalid_argument& e) {
    return -1;
  } catch (std::out_of_range& e) {
    return -1;
  }
  if (col < 0 || col >= kBoardDimension || row < 0 || row >= kBoardDimension) {
    return -1;
  }

  return row * kBoardDimension + col;
}

}  // namespace othello
