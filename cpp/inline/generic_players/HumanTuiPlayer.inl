#include <generic_players/HumanTuiPlayer.hpp>

#include <cstdlib>
#include <iostream>

#include <core/DerivedTypes.hpp>
#include <util/ScreenUtil.hpp>

namespace generic {

template <core::GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::start_game() {
  GameStateTypes::nullify_action(last_action_);
  std::cout << "Press any key to start game" << std::endl;
  std::string input;
  std::getline(std::cin, input);

  util::clearscreen();
}

template <core::GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::receive_state_change(core::seat_index_t, const GameState&,
                                                             const Action& action) {
  last_action_ = action;
}

template <core::GameStateConcept GameState_>
typename HumanTuiPlayer<GameState_>::ActionResponse HumanTuiPlayer<GameState_>::get_action_response(
    const GameState& state, const ActionMask& valid_actions) {
  util::ScreenClearer::clear_once();
  print_state(state, false);
  bool complain = false;
  Action my_action;
  GameStateTypes::nullify_action(my_action);
  while (true) {
    if (complain) {
      printf("Invalid input!\n");
    }
    complain = true;
    my_action = prompt_for_action(state, valid_actions);

    if (!GameStateTypes::is_valid_action(my_action, valid_actions)) continue;
    break;
  }

  util::ScreenClearer::reset();
  return my_action;
}

template <core::GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::end_game(const GameState& state,
                                                 const GameOutcome& outcome) {
  util::ScreenClearer::clear_once();
  print_state(state, true);

  auto seat = this->get_my_seat();
  if (outcome[seat] == 1) {
    std::cout << "Congratulations, you win!" << std::endl;
  } else if (outcome[1 - seat] == 1) {
    std::cout << "Sorry, you lose." << std::endl;
  } else {
    std::cout << "The game has ended in a draw." << std::endl;
  }
}

template <core::GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::print_state(const GameState& state, bool terminal) {
  state.dump(&last_action_, &this->get_player_names());
}

}  // namespace generic
