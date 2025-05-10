#include <generic_players/HumanTuiPlayer.hpp>

#include <util/ScreenUtil.hpp>

#include <cstdlib>
#include <iostream>

namespace generic {

template <core::concepts::Game Game>
inline void HumanTuiPlayer<Game>::start_game() {
  last_action_ = -1;
  std::cout << "Press any key to start game" << std::endl;
  std::string input;
  std::getline(std::cin, input);

  util::clearscreen();
}

template <core::concepts::Game Game>
inline void HumanTuiPlayer<Game>::receive_state_change(core::seat_index_t, const State&,
                                                       core::action_t action) {
  last_action_ = action;
}

// TODO: return a core::kYield, and do the std::cin handling in a separate thread
template <core::concepts::Game Game>
typename HumanTuiPlayer<Game>::ActionResponse HumanTuiPlayer<Game>::get_action_response(
    const ActionRequest& request) {
  const State& state = request.state;
  const ActionMask& valid_actions = request.valid_actions;

  util::ScreenClearer::clear_once();
  print_state(state, false);
  bool complain = false;
  core::action_t my_action = -1;
  while (true) {
    if (complain) {
      printf("Invalid input!\n");
    }
    complain = true;
    my_action = prompt_for_action(state, valid_actions);

    if (my_action < 0 || my_action >= Game::Types::kMaxNumActions || !valid_actions[my_action]) {
      continue;
    }
    break;
  }

  util::ScreenClearer::reset();
  return my_action;
}

template <core::concepts::Game Game>
inline void HumanTuiPlayer<Game>::end_game(const State& state, const ValueTensor& outcome) {
  util::ScreenClearer::clear_once();
  print_state(state, true);

  auto array = Game::GameResults::to_value_array(outcome);
  auto seat = this->get_my_seat();
  if (array[seat] == 1) {
    std::cout << "Congratulations, you win!" << std::endl;
  } else if (array[seat] == 0) {
    std::cout << "Sorry, you lose." << std::endl;
  } else {
    std::cout << "The game has ended in a draw." << std::endl;
  }
}

template <core::concepts::Game Game>
inline void HumanTuiPlayer<Game>::print_state(const State& state, bool terminal) {
  IO::print_state(std::cout, state, last_action_, &this->get_player_names());
}

}  // namespace generic
