#include <games/othello/Game.hpp>

#include <algorithm>
#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <util/AnsiCodes.hpp>
#include <util/BitMapUtil.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

namespace othello {

inline size_t Game::BaseState::hash() const {
  return util::tuple_hash(std::make_tuple(opponent_mask, cur_player_mask, cur_player, pass_count));
}

inline void Game::Symmetries::apply(BaseState& state, const core::symmetry_t& sym) {
  util::release_assert(sym.group_id == 0, "Unknown group: %d", sym.group_id);
  switch (sym.element) {
    case groups::D4::kIdentity:
      return;
    case groups::D4::kRot90: {
      bitmap_util::rot90_clockwise(state.cur_player_mask);
      bitmap_util::rot90_clockwise(state.opponent_mask);
      return;
    }
    case groups::D4::kRot180: {
      bitmap_util::rot180(state.cur_player_mask);
      bitmap_util::rot180(state.opponent_mask);
      return;
    }
    case groups::D4::kRot270: {
      bitmap_util::rot270_clockwise(state.cur_player_mask);
      bitmap_util::rot270_clockwise(state.opponent_mask);
      return;
    }
    case groups::D4::kFlipVertical: {
      bitmap_util::flip_vertical(state.cur_player_mask);
      bitmap_util::flip_vertical(state.opponent_mask);
      return;
    }
    case groups::D4::kFlipMainDiag: {
      bitmap_util::flip_main_diag(state.cur_player_mask);
      bitmap_util::flip_main_diag(state.opponent_mask);
      return;
    }
    case groups::D4::kMirrorHorizontal: {
      bitmap_util::mirror_horizontal(state.cur_player_mask);
      bitmap_util::mirror_horizontal(state.opponent_mask);
      return;
    }
    case groups::D4::kFlipAntiDiag: {
      bitmap_util::flip_anti_diag(state.cur_player_mask);
      bitmap_util::flip_anti_diag(state.opponent_mask);
      return;
    }
    default: {
      throw util::Exception("Unknown group element: %d", sym.element);
    }
  }
}

inline void Game::Symmetries::apply(Types::PolicyTensor& tensor, const core::symmetry_t& sym) {
  util::release_assert(sym.group_id == 0, "Unknown group: %d", sym.group_id);
  switch (sym.element) {
    case groups::D4::kIdentity:
      return;
    case groups::D4::kRot90: {
      eigen_util::rot90_clockwise<kBoardDimension>(tensor);
      return;
    }
    case groups::D4::kRot180: {
      eigen_util::rot180<kBoardDimension>(tensor);
      return;
    }
    case groups::D4::kRot270: {
      eigen_util::rot270_clockwise<kBoardDimension>(tensor);
      return;
    }
    case groups::D4::kFlipVertical: {
      eigen_util::flip_vertical<kBoardDimension>(tensor);
      return;
    }
    case groups::D4::kFlipMainDiag: {
      eigen_util::flip_main_diag<kBoardDimension>(tensor);
      return;
    }
    case groups::D4::kMirrorHorizontal: {
      eigen_util::mirror_horizontal<kBoardDimension>(tensor);
      return;
    }
    case groups::D4::kFlipAntiDiag: {
      eigen_util::flip_anti_diag<kBoardDimension>(tensor);
      return;
    }
    default: {
      throw util::Exception("Unknown group element: %d", sym.element);
    }
  }
}

inline void Game::Rules::init_state(FullState& state) {
  state.opponent_mask = kStartingWhiteMask;
  state.cur_player_mask = kStartingBlackMask;
  state.cur_player = kStartingColor;
  state.pass_count = 0;
}

inline core::seat_index_t Game::Rules::get_current_player(const BaseState& state) {
  return state.cur_player;
}

inline Game::InputTensorizor::Tensor Game::InputTensorizor::tensorize(const BaseState* start,
                                                                      const BaseState* cur) {
  const BaseState& state = *cur;
  core::seat_index_t cp = Rules::get_current_player(state);
  Tensor tensor;
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      core::seat_index_t p = get_player_at(state, row, col);
      tensor(0, row, col) = (p == cp);
      tensor(1, row, col) = (p == 1 - cp);
    }
  }
  return tensor;
}

inline Game::TrainingTargets::ScoreMarginTarget::Tensor
Game::TrainingTargets::ScoreMarginTarget::tensorize(const Types::GameLogView& view) {
  Tensor tensor;
  const BaseState& state = *view.cur_pos;
  core::seat_index_t cp = Rules::get_current_player(state);
  tensor(0) = get_count(state, cp) - get_count(state, 1 - cp);
  return tensor;
}

inline Game::TrainingTargets::OwnershipTarget::Tensor
Game::TrainingTargets::OwnershipTarget::tensorize(const Types::GameLogView& view) {
  Tensor tensor;
  const BaseState& cur_state = *view.cur_pos;
  const BaseState& final_state = *view.final_pos;
  core::seat_index_t cp = Rules::get_current_player(cur_state);
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      core::seat_index_t p = get_player_at(final_state, row, col);
      int val = (p == -1) ? 0 : ((p == cp) ? 2 : 1);
      tensor(row, col) = val;
    }
  }
  return tensor;
}

inline int Game::get_count(const BaseState& state, core::seat_index_t seat) {
  if (seat == state.cur_player) {
    return std::popcount(state.cur_player_mask);
  } else {
    return std::popcount(state.opponent_mask);
  }
}

inline core::seat_index_t Game::get_player_at(const BaseState& state, int row, int col) {
  int cp = Rules::get_current_player(state);
  int index = row * kBoardDimension + col;
  bool occupied_by_cur_player = (mask_t(1) << index) & state.cur_player_mask;
  bool occupied_by_opponent = (mask_t(1) << index) & state.opponent_mask;
  return occupied_by_opponent ? (1 - cp) : (occupied_by_cur_player ? cp : -1);
}

// copied from edax-reversi repo
inline mask_t Game::get_moves(mask_t P, mask_t O) {
  mask_t mask = O & 0x7E7E7E7E7E7E7E7Eull;

  return (get_some_moves(P, mask, 1)    // horizontal
          | get_some_moves(P, O, 8)     // vertical
          | get_some_moves(P, mask, 7)  // diagonals
          | get_some_moves(P, mask, 9)) &
         ~(P | O);  // mask with empties
}

// copied from edax-reversi repo
inline mask_t Game::get_some_moves(mask_t P, mask_t mask, int dir) {
#if PARALLEL_PREFIX & 1
  // 1-stage Parallel Prefix (intermediate between kogge stone & sequential)
  // 6 << + 6 >> + 7 | + 10 &
  register unsigned long long flip_l, flip_r;
  register unsigned long long mask_l, mask_r;
  const int dir2 = dir + dir;

  flip_l = mask & (P << dir);
  flip_r = mask & (P >> dir);
  flip_l |= mask & (flip_l << dir);
  flip_r |= mask & (flip_r >> dir);
  mask_l = mask & (mask << dir);
  mask_r = mask & (mask >> dir);
  flip_l |= mask_l & (flip_l << dir2);
  flip_r |= mask_r & (flip_r >> dir2);
  flip_l |= mask_l & (flip_l << dir2);
  flip_r |= mask_r & (flip_r >> dir2);

  return (flip_l << dir) | (flip_r >> dir);

#elif KOGGE_STONE & 1
  // kogge-stone algorithm
  // 6 << + 6 >> + 12 & + 7 |
  // + better instruction independency
  register unsigned long long flip_l, flip_r;
  register unsigned long long mask_l, mask_r;
  const int dir2 = dir << 1;
  const int dir4 = dir << 2;

  flip_l = P | (mask & (P << dir));
  flip_r = P | (mask & (P >> dir));
  mask_l = mask & (mask << dir);
  mask_r = mask & (mask >> dir);
  flip_l |= mask_l & (flip_l << dir2);
  flip_r |= mask_r & (flip_r >> dir2);
  mask_l &= (mask_l << dir2);
  mask_r &= (mask_r >> dir2);
  flip_l |= mask_l & (flip_l << dir4);
  flip_r |= mask_r & (flip_r >> dir4);

  return ((flip_l & mask) << dir) | ((flip_r & mask) >> dir);

#else
  // sequential algorithm
  // 7 << + 7 >> + 6 & + 12 |
  mask_t flip;

  flip = (((P << dir) | (P >> dir)) & mask);
  flip |= (((flip << dir) | (flip >> dir)) & mask);
  flip |= (((flip << dir) | (flip >> dir)) & mask);
  flip |= (((flip << dir) | (flip >> dir)) & mask);
  flip |= (((flip << dir) | (flip >> dir)) & mask);
  flip |= (((flip << dir) | (flip >> dir)) & mask);
  return (flip << dir) | (flip >> dir);

#endif
}

}  // namespace othello
