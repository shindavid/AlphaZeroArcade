#include "games/tictactoe/players/WebPlayer.hpp"

namespace tictactoe {

void WebPlayer::start_game() { std::cout << "Web player started the game." << std::endl; }

void WebPlayer::receive_state_change(core::seat_index_t, const State& state,
                                     core::action_t action) {
  std::cout << "Web player received state change: " << IO::compact_state_repr(state)
            << " after action: " << action << std::endl;
}

typename WebPlayer::ActionResponse WebPlayer::get_action_response(
  const ActionRequest& request) {
  std::cout << "Web player is making a move." << std::endl;
  // Here you would typically send the request to the frontend and wait for a response.
  // For now, we will just return a dummy action.
  return 0;  // Replace with actual logic to get the action from the user.
}

void WebPlayer::end_game(const State& state, const ValueTensor& outcome) {
  std::cout << "Web player ended the game. Final state: " << IO::compact_state_repr(state)
            << " with outcome: " << outcome << std::endl;
  auto array = Game::GameResults::to_value_array(outcome);
  auto seat = this->get_my_seat();
  if (array[seat] == 1) {
    std::cout << "Web player won!" << std::endl;
  } else if (array[seat] == 0) {
    std::cout << "Web player lost." << std::endl;
  } else {
    std::cout << "Web player drew." << std::endl;
  }
}

}  // namespace tictactoe
