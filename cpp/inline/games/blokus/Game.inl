#include "games/blokus/Game.hpp"

#include <format>

namespace blokus {

inline core::action_mode_t Game::Rules::get_action_mode(const State& state) {
  if (state.core.partial_move.valid()) {
    return kPiecePlacementMode;
  } else {
    return kLocationMode;
  }
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.core.cur_color;
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t mode) {
  if (mode == kLocationMode) {
    if (action == kPass) {
      return "pass";
    }
    Location loc = Location::unflatten(action);
    int row = loc.row + 1;
    char col = 'A' + loc.col;
    return std::format("{}{}", col, row);
  } else {
    PieceOrientationCorner poc = PieceOrientationCorner::from_action(action);
    return poc.name();
  }
}

}  // namespace blokus
