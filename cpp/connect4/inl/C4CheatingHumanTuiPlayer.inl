#include <connect4/C4CheatingHumanTuiPlayer.hpp>

namespace c4 {

inline CheatingHumanTuiPlayer::CheatingHumanTuiPlayer(const PerfectPlayParams& perfect_play_params)
: base_t(), oracle_(perfect_play_params) {}

inline void CheatingHumanTuiPlayer::start_game() {
  move_history_.reset();
  base_t::start_game();
}

inline void CheatingHumanTuiPlayer::receive_state_change(
    common::player_index_t player, const GameState& state, common::action_index_t action, const GameOutcome& outcome) {
  move_history_.append(action);
  base_t::receive_state_change(player, state, action, outcome);
}

inline void CheatingHumanTuiPlayer::print_state(const GameState& state) {
  auto result = oracle_.query(move_history_);
  printf("%s\n", result.get_overlay().c_str());
  state.dump(last_action_, &player_names_);
}

}  // namespace c4
