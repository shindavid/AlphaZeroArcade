#include <games/blokus/Game.hpp>

#include <util/CppUtil.hpp>
#include <util/StringUtil.hpp>

namespace blokus {

inline size_t Game::State::hash() const {
  return util::PODHash<core_t>{}(core);
}

inline int Game::State::remaining_square_count(color_t c) const {
  return kNumSquaresPerColor - core.occupied_locations[c].count();
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
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
    return util::create_string("%c%d", col, row);
  }

  PieceOrientationCorner poc = PieceOrientationCorner::from_action(action);
  return poc.name();
}

template <typename Iter>
Game::InputTensorizor::Tensor Game::InputTensorizor::tensorize(Iter start, Iter cur) {
  core::seat_index_t cp = Rules::get_current_player(*cur);
  Tensor tensor;
  tensor.setZero();
  int i = 0;
  Iter state = cur;
  while (true) {
    for (color_t c = 0; c < kNumColors; ++c) {
      color_t rc = (kNumColors + c - cp) % kNumColors;
      for (Location loc : state->core.occupied_locations[c].get_set_locations()) {
        tensor(i + rc, loc.row, loc.col) = 1;
      }
    }
    if (state == start) break;
    state--;
    i += kNumColors;
  }

  if (cur->core.partial_move.valid()) {
    Location loc = cur->core.partial_move;
    tensor(kDim0 - 1, loc.row, loc.col) = 1;
  }
  return tensor;
}

}  // namespace blokus
