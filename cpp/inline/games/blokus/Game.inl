#include <games/blokus/Game.hpp>

#include <util/CppUtil.hpp>

namespace blokus {

inline size_t Game::BaseState::hash() const {
  return util::PODHash<core_t>{}(core);
}

inline int Game::BaseState::remaining_square_count(color_t c) const {
  return kNumSquaresPerColor - core.occupied_locations[c].count();
}

inline Game::Types::SymmetryMask Game::Symmetries::get_mask(const BaseState& state) {
  Types::SymmetryMask mask;
  mask.set();
  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const BaseState& state) {
  return state.core.cur_color;
}

inline std::string Game::IO::action_to_str(core::action_t action) {
  if (action == kPass) {
    return "pass";
  }
  if (action < kPass) {
    Location loc = Location::unflatten(action);
    int row = loc.row + 1;
    char col = 'A' + loc.col;
    return std::string(1, col) + std::to_string(row);
  }

  PieceOrientationCorner poc = PieceOrientationCorner::from_action(action);
  return poc.name();
}

}  // namespace blokus
