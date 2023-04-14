#include <common/HumanTuiPlayer.hpp>

#include <cstdlib>
#include <iostream>

#include <common/DerivedTypes.hpp>
#include <util/ScreenUtil.hpp>

namespace common {

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::start_game() {
  std::cout << "Press any key to start game" << std::endl;
  std::string input;
  std::getline(std::cin, input);

  util::clearscreen();
}

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::receive_state_change(
    common::seat_index_t, const GameState&, common::action_index_t action)
{
  last_action_ = action;
}

template<GameStateConcept GameState_>
inline common::action_index_t HumanTuiPlayer<GameState_>::get_action(
    const GameState& state, const ActionMask& valid_actions)
{
  util::ScreenClearer::clear_once();
  print_state(state, false);
  bool complain = false;
  int my_action = -1;
  while (true) {
    if (complain) {
      printf("Invalid input!\n");
    }
    complain = true;
    try {
      my_action = GameState::prompt_for_action();
      if (!valid_actions.test(my_action)) continue;
    } catch(...) {
      continue;
    }
    break;
  }

  util::ScreenClearer::reset();
  return my_action;
}

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::end_game(const GameState& state, const GameOutcome& outcome) {
  util::ScreenClearer::clear_once();
  print_state(state, true);

  auto seat = this->get_my_seat();
  if (outcome[seat] == 1) {
    std::cout << "Congratulations, you win!" << std::endl;
  } else if (outcome[1-seat] == 1) {
    std::cout << "Sorry, you lose." << std::endl;
  } else {
    std::cout << "The game has ended in a draw." << std::endl;
  }
}

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::print_state(const GameState& state, bool terminal) {
  state.dump(last_action_, &this->get_player_names());
}

}  // namespace common
