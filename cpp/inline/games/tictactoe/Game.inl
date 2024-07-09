#include <games/tictactoe/Game.hpp>

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

inline size_t Game::BaseState::hash() const {
  return (size_t(full_mask) << 16) + cur_player_mask;
}

inline Game::Types::SymmetryMask Game::Symmetries::get_mask(const BaseState& state) {
  Types::SymmetryMask mask;
  mask.set();
  return mask;
}

inline void Game::Symmetries::apply(BaseState& state, group::element_t sym) {
  using namespace tictactoe::detail;
  using D4 = groups::D4;
  auto& s = state;
  switch (sym) {
    case D4::kIdentity: return;
    case D4::kRot90: return rot90_clockwise(s.cur_player_mask, s.full_mask);
    case D4::kRot180: return rot180(s.cur_player_mask, s.full_mask);
    case D4::kRot270: return rot270_clockwise(s.cur_player_mask, s.full_mask);
    case D4::kFlipVertical: return flip_vertical(s.cur_player_mask, s.full_mask);
    case D4::kFlipMainDiag: return flip_main_diag(s.cur_player_mask, s.full_mask);
    case D4::kMirrorHorizontal: return mirror_horizontal(s.cur_player_mask, s.full_mask);
    case D4::kFlipAntiDiag: return flip_anti_diag(s.cur_player_mask, s.full_mask);
    default: {
      throw util::Exception("Unknown group element: %d", sym);
    }
  }
}

inline void Game::Symmetries::apply(Types::PolicyTensor& tensor, group::element_t sym) {
  using namespace eigen_util;
  using D4 = groups::D4;
  constexpr int N = kBoardDimension;
  switch (sym) {
    case D4::kIdentity: return;
    case D4::kRot90: return rot90_clockwise<N>(tensor);
    case D4::kRot180: return rot180<N>(tensor);
    case D4::kRot270: return rot270_clockwise<N>(tensor);
    case D4::kFlipVertical: return flip_vertical<N>(tensor);
    case D4::kFlipMainDiag: return flip_main_diag<N>(tensor);
    case D4::kMirrorHorizontal: return mirror_horizontal<N>(tensor);
    case D4::kFlipAntiDiag: return flip_anti_diag<N>(tensor);
    default: {
      throw util::Exception("Unknown group element: %d", sym);
    }
  }
}

inline void Game::Rules::init_state(FullState& state) {
  state.full_mask = 0;
  state.cur_player_mask = 0;
}

inline core::seat_index_t Game::Rules::get_current_player(const BaseState& state) {
  return std::popcount(state.full_mask) % 2;
}

inline Game::InputTensorizor::Tensor Game::InputTensorizor::tensorize(const BaseState* start,
                                                                      const BaseState* cur) {
  core::seat_index_t cp = Rules::get_current_player(*cur);
  Tensor tensor;
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      core::seat_index_t p = _get_player_at(*cur, row, col);
      tensor(0, row, col) = (p == cp);
      tensor(1, row, col) = (p == 1 - cp);
    }
  }
  return tensor;
}

inline Game::TrainingTargets::OwnershipTarget::Tensor
Game::TrainingTargets::OwnershipTarget::tensorize(const Types::GameLogView& view) {
  Tensor tensor;
  const BaseState& state = *view.cur_pos;
  core::seat_index_t cp = Rules::get_current_player(state);
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      core::seat_index_t p = _get_player_at(state, row, col);
      int val = (p == -1) ? 0 : ((p == cp) ? 2 : 1);
      tensor(row, col) = val;
    }
  }
  return tensor;
}

inline core::seat_index_t Game::_get_player_at(const BaseState& state, int row, int col) {
  int cp = Rules::get_current_player(state);
  int index = row * kBoardDimension + col;
  bool occupied_by_cur_player = (mask_t(1) << index) & state.cur_player_mask;
  bool occupied_by_any_player = (mask_t(1) << index) & state.full_mask;
  return occupied_by_any_player ? (occupied_by_cur_player ? cp : (1 - cp)) : -1;
}

}  // namespace tictactoe
