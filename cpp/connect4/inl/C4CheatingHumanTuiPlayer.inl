#include <connect4/C4CheatingHumanTuiPlayer.hpp>

namespace c4 {

inline void CheatingHumanTuiPlayer::start_game() {
  move_history_.reset();
  base_t::start_game();
}

inline void CheatingHumanTuiPlayer::receive_state_change(
    common::player_index_t player, const GameState& state, common::action_index_t action) {
  move_history_.append(action);
  base_t::receive_state_change(player, state, action);
}

inline void CheatingHumanTuiPlayer::end_game(const GameState& state, const GameOutcome&) {
  util::ScreenClearer::clear_once();
  std::cout << std::endl;  // blank link to match print_state()
  base_t::print_state(state);
}

inline void CheatingHumanTuiPlayer::print_state(const GameState& state) {
  auto result = oracle_.query(move_history_);
  printf("%s\n", result.get_overlay().c_str());
  state.dump(last_action_, &this->get_player_names());
}

}  // namespace c4
