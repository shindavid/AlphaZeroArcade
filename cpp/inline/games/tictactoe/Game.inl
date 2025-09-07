#include "games/tictactoe/Game.hpp"

#include "core/DefaultCanonicalizer.hpp"

namespace tictactoe {

namespace detail {

inline void flip_vertical(mask_t& mask) {
  mask = ((mask & 0x07) << 6) | ((mask & 0x38) << 0) | ((mask & 0x1c0) >> 6);
}

inline void flip_main_diag(mask_t& mask) {
  mask = (mask & 0x111) | ((mask & 0x22) << 2) | ((mask & 0x4) << 4) | ((mask & 0x88) >> 2) |
         ((mask & 0x40) >> 4);
}

inline void mirror_horizontal(mask_t& mask) {
  mask = ((mask & 0x49) << 2) | ((mask & 0x92) << 0) | ((mask & 0x124) >> 2);
}

inline void flip_anti_diag(mask_t& mask) {
  mask = (mask & 0x54) | ((mask & 0x100) >> 8) | ((mask & 0x01) << 8) | ((mask & 0x0a) << 4) |
         ((mask & 0xa0) >> 4);
}

inline void rot90_clockwise(mask_t& mask) {
  flip_vertical(mask);
  flip_main_diag(mask);
}

inline void rot180(mask_t& mask) {
  flip_vertical(mask);
  mirror_horizontal(mask);
}

inline void rot270_clockwise(mask_t& mask) {
  flip_vertical(mask);
  flip_anti_diag(mask);
}

inline void rot90_clockwise(mask_t& mask1, mask_t& mask2) {
  rot90_clockwise(mask1);
  rot90_clockwise(mask2);
}

inline void rot180(mask_t& mask1, mask_t& mask2) {
  rot180(mask1);
  rot180(mask2);
}

inline void rot270_clockwise(mask_t& mask1, mask_t& mask2) {
  rot270_clockwise(mask1);
  rot270_clockwise(mask2);
}

inline void flip_vertical(mask_t& mask1, mask_t& mask2) {
  flip_vertical(mask1);
  flip_vertical(mask2);
}

inline void flip_main_diag(mask_t& mask1, mask_t& mask2) {
  flip_main_diag(mask1);
  flip_main_diag(mask2);
}

inline void mirror_horizontal(mask_t& mask1, mask_t& mask2) {
  mirror_horizontal(mask1);
  mirror_horizontal(mask2);
}

inline void flip_anti_diag(mask_t& mask1, mask_t& mask2) {
  flip_anti_diag(mask1);
  flip_anti_diag(mask2);
}

}  // namespace detail

inline size_t Game::State::hash() const { return (size_t(full_mask) << 16) + cur_player_mask; }

inline core::seat_index_t Game::State::get_player_at(int row, int col) const {
  int cp = Rules::get_current_player(*this);
  int index = row * kBoardDimension + col;
  bool occupied_by_cur_player = (mask_t(1) << index) & cur_player_mask;
  bool occupied_by_any_player = (mask_t(1) << index) & full_mask;
  return occupied_by_any_player ? (occupied_by_cur_player ? cp : (1 - cp)) : -1;
}

inline Game::Types::SymmetryMask Game::Symmetries::get_mask(const State& state) {
  Types::SymmetryMask mask;
  mask.set();
  return mask;
}

inline void Game::Symmetries::apply(State& state, group::element_t sym) {
  using namespace tictactoe::detail;
  using D4 = groups::D4;
  auto& s = state;
  switch (sym) {
    case D4::kIdentity:
      return;
    case D4::kRot90:
      return rot90_clockwise(s.cur_player_mask, s.full_mask);
    case D4::kRot180:
      return rot180(s.cur_player_mask, s.full_mask);
    case D4::kRot270:
      return rot270_clockwise(s.cur_player_mask, s.full_mask);
    case D4::kFlipVertical:
      return flip_vertical(s.cur_player_mask, s.full_mask);
    case D4::kFlipMainDiag:
      return flip_main_diag(s.cur_player_mask, s.full_mask);
    case D4::kMirrorHorizontal:
      return mirror_horizontal(s.cur_player_mask, s.full_mask);
    case D4::kFlipAntiDiag:
      return flip_anti_diag(s.cur_player_mask, s.full_mask);
    default: {
      throw util::Exception("Unknown group element: {}", sym);
    }
  }
}

inline void Game::Symmetries::apply(StateHistory& history, group::element_t sym) {
  for (auto& it : history) {
    apply(it, sym);
  }
}

inline void Game::Symmetries::apply(Types::PolicyTensor& tensor, group::element_t sym,
                                    core::action_mode_t) {
  using namespace eigen_util;
  using D4 = groups::D4;
  constexpr int N = kBoardDimension;
  switch (sym) {
    case D4::kIdentity:
      return;
    case D4::kRot90:
      return rot90_clockwise<N>(tensor);
    case D4::kRot180:
      return rot180<N>(tensor);
    case D4::kRot270:
      return rot270_clockwise<N>(tensor);
    case D4::kFlipVertical:
      return flip_vertical<N>(tensor);
    case D4::kFlipMainDiag:
      return flip_main_diag<N>(tensor);
    case D4::kMirrorHorizontal:
      return mirror_horizontal<N>(tensor);
    case D4::kFlipAntiDiag:
      return flip_anti_diag<N>(tensor);
    default: {
      throw util::Exception("Unknown group element: {}", sym);
    }
  }
}

inline void Game::Symmetries::apply(core::action_t& action, group::element_t sym,
                                    core::action_mode_t) {
  constexpr int8_t lookup[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8,  // kIdentity
    2, 5, 8, 1, 4, 7, 0, 3, 6,  // kRot90
    8, 7, 6, 5, 4, 3, 2, 1, 0,  // kRot180
    6, 3, 0, 7, 4, 1, 8, 5, 2,  // kRot270
    6, 7, 8, 3, 4, 5, 0, 1, 2,  // kFlipVertical
    0, 3, 6, 1, 4, 7, 2, 5, 8,  // kFlipMainDiag
    2, 1, 0, 5, 4, 3, 8, 7, 6,  // kMirrorHorizontal
    8, 5, 2, 7, 4, 1, 6, 3, 0,  // kFlipAntiDiag
  };

  action = lookup[sym * 9 + action];
}

inline group::element_t Game::Symmetries::get_canonical_symmetry(const State& state) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<Game>;
  return DefaultCanonicalizer::get(state);
}

inline void Game::Rules::init_state(State& state) {
  state.full_mask = 0;
  state.cur_player_mask = 0;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return std::popcount(state.full_mask) % 2;
}

inline std::string Game::IO::compact_state_repr(const State& state) {
  char buf[12];
  const char* syms = "_XO";

  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      buf[row * 4 + col] = syms[state.get_player_at(row, col) + 1];
    }
  }
  buf[3] = '\n';
  buf[7] = '\n';
  buf[11] = '\0';

  return std::string(buf);
}

inline boost::json::value Game::IO::state_to_json(const State& state) {
  char buf[10];
  const char* syms = "_XO";

  int c = 0;
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      buf[c++] = syms[state.get_player_at(row, col) + 1];
    }
  }
  buf[c] = '\0';
  return boost::json::value(std::string(buf));
}

}  // namespace tictactoe
