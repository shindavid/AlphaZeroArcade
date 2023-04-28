#include <connect4/players/C4CheatingHumanTuiPlayer.hpp>

namespace c4 {

inline void CheatingHumanTuiPlayer::start_game() {
  move_history_.reset();
  base_t::start_game();
}

inline void CheatingHumanTuiPlayer::receive_state_change(
    common::seat_index_t seat, const GameState& state, common::action_index_t action) {
  move_history_.append(action);
  base_t::receive_state_change(seat, state, action);
}

inline void CheatingHumanTuiPlayer::print_state(const GameState& state, bool terminal) {
  if (terminal) {
    std::cout << std::endl;  // blank link to retain alignment
  } else {
    auto result = oracle_.query(move_history_);
    printf("%s\n", result.get_overlay().c_str());
  }
  state.dump(last_action_, &this->get_player_names());
}

}  // namespace c4
