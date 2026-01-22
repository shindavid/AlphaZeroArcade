#include "generic_players/HumanTuiPlayer.hpp"

#include "core/BasicTypes.hpp"
#include "search/VerboseManager.hpp"
#include "util/ScreenUtil.hpp"

#include <cstdlib>
#include <iostream>

namespace generic {

template <core::concepts::Game Game>
inline bool HumanTuiPlayer<Game>::start_game() {
  last_action_ = -1;
  std::cout << "Press any key to start game" << std::endl;
  std::string input;
  std::getline(std::cin, input);

  util::clearscreen();
  return true;
}

template <core::concepts::Game Game>
inline void HumanTuiPlayer<Game>::receive_state_change(const StateChangeUpdate& update) {
  last_action_ = update.action();
}

// TODO: return a core::kYield, and do the std::cin handling in a separate thread
template <core::concepts::Game Game>
core::ActionResponse HumanTuiPlayer<Game>::get_action_response(const ActionRequest& request) {
  util::clearscreen();
  print_state(request.state, false);

  auto verbose_data = request.verbose_data_iterator.most_recent_data();
  if (verbose_data) {
    verbose_data->to_terminal_text();
  }

  bool complain = false;
  while (true) {
    if (complain) {
      printf("Invalid input!\n");
    }
    complain = true;
    auto response = prompt_for_action(request);
    if (!request.permits(response)) continue;
    return response;
  }
}

template <core::concepts::Game Game>
inline void HumanTuiPlayer<Game>::end_game(const State& state, const GameResultTensor& outcome) {
  util::clearscreen();
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
