#include <games/othello/Game.hpp>

#include <core/DefaultCanonicalizer.hpp>
#include <util/AnsiCodes.hpp>
#include <util/BitMapUtil.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

#include <algorithm>
#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

namespace othello {

inline size_t Game::State::hash() const {
  auto tuple = std::make_tuple(opponent_mask, cur_player_mask, cur_player, pass_count);
  std::hash<decltype(tuple)> hasher;
  return hasher(tuple);
}

inline Game::Types::SymmetryMask Game::Symmetries::get_mask(const State& state) {
  Types::SymmetryMask mask;
  mask.set();
  return mask;
}

inline void Game::Symmetries::apply(State& state, group::element_t sym) {
  using namespace bitmap_util;
  using D4 = groups::D4;
  auto& s = state;
  switch (sym) {
    case D4::kIdentity: return;
    case D4::kRot90: return rot90_clockwise(s.cur_player_mask, s.opponent_mask);
    case D4::kRot180: return rot180(s.cur_player_mask, s.opponent_mask);
    case D4::kRot270: return rot270_clockwise(s.cur_player_mask, s.opponent_mask);
    case D4::kFlipVertical: return flip_vertical(s.cur_player_mask, s.opponent_mask);
    case D4::kFlipMainDiag: return flip_main_diag(s.cur_player_mask, s.opponent_mask);
    case D4::kMirrorHorizontal: return mirror_horizontal(s.cur_player_mask, s.opponent_mask);
    case D4::kFlipAntiDiag: return flip_anti_diag(s.cur_player_mask, s.opponent_mask);
    default: throw util::Exception("Unknown group element: %d", sym);
  }
}

inline void Game::Symmetries::apply(StateHistory& history, group::element_t sym) {
  for (auto& it : history) {
    apply(it, sym);
  }
}

inline void Game::Symmetries::apply(Types::PolicyTensorVariant& policy_variant, group::element_t sym) {
  using namespace eigen_util;
  using D4 = groups::D4;
  constexpr int N = kBoardDimension;

  auto& tensor = std::get<0>(policy_variant);
  switch (sym) {
    case D4::kIdentity: return;
    case D4::kRot90: return rot90_clockwise<N>(tensor);
    case D4::kRot180: return rot180<N>(tensor);
    case D4::kRot270: return rot270_clockwise<N>(tensor);
    case D4::kFlipVertical: return flip_vertical<N>(tensor);
    case D4::kFlipMainDiag: return flip_main_diag<N>(tensor);
    case D4::kMirrorHorizontal: return mirror_horizontal<N>(tensor);
    case D4::kFlipAntiDiag: return flip_anti_diag<N>(tensor);
    default: throw util::Exception("Unknown group element: %d", sym);
  }
}

inline void Game::Symmetries::apply(core::action_t& action, group::element_t sym) {
  using namespace bitmap_util;
  using D4 = groups::D4;

  if (action == kPass || sym == D4::kIdentity) return;

  mask_t mask = 1ULL << action;

  switch (sym) {
    case D4::kRot90: rot90_clockwise(mask); break;
    case D4::kRot180: rot180(mask); break;
    case D4::kRot270: rot270_clockwise(mask); break;
    case D4::kFlipVertical: flip_vertical(mask); break;
    case D4::kFlipMainDiag: flip_main_diag(mask); break;
    case D4::kMirrorHorizontal: mirror_horizontal(mask); break;
    case D4::kFlipAntiDiag: flip_anti_diag(mask); break;
    default: throw util::Exception("Unknown group element: %d", sym);
  }

  action = std::countr_zero(mask);
}

inline group::element_t Game::Symmetries::get_canonical_symmetry(const State& state) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<Game>;
  return DefaultCanonicalizer::get(state);
}

inline void Game::Rules::init_state(State& state) {
  state.opponent_mask = kStartingWhiteMask;
  state.cur_player_mask = kStartingBlackMask;
  state.cur_player = kStartingColor;
  state.pass_count = 0;
}

inline Game::Types::ActionMaskVariant Game::Rules::get_legal_moves(const StateHistory& history) {
  return get_legal_moves(history.current());
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.cur_player;
}

template <typename Iter>
Game::InputTensorizor::Tensor Game::InputTensorizor::tensorize(Iter start, Iter cur) {
  core::seat_index_t cp = Rules::get_current_player(*cur);
  Tensor tensor;
  tensor.setZero();
  int i = 0;
  Iter state = cur;
  while (true) {
    for (int row = 0; row < kBoardDimension; ++row) {
      for (int col = 0; col < kBoardDimension; ++col) {
        core::seat_index_t p = get_player_at(*state, row, col);
        if (p < 0) continue;
        int x = (Constants::kNumPlayers + cp - p) % Constants::kNumPlayers;
        tensor(i + x, row, col) = 1;
      }
    }
    if (state == start) break;
    state--;
    i += kNumPlayers;
  }
  return tensor;
}

inline Game::TrainingTargets::ScoreMarginTarget::Tensor
Game::TrainingTargets::ScoreMarginTarget::tensorize(const Types::GameLogView& view) {
  Tensor tensor;
  tensor.setZero();
  const State& state = *view.final_pos;
  core::seat_index_t cp = Rules::get_current_player(*view.cur_pos);
  int score_index = kNumCells + get_count(state, cp) - get_count(state, 1 - cp);
  util::release_assert(score_index >= 0 && score_index <= kNumCells * 2);

  // PDF
  tensor(0, score_index) = 1;

  // CDF
  for (int i = 0; i <= score_index; ++i) {
    tensor(1, i) = 1;
  }
  return tensor;
}

inline Game::TrainingTargets::OwnershipTarget::Tensor
Game::TrainingTargets::OwnershipTarget::tensorize(const Types::GameLogView& view) {
  Tensor tensor;
  tensor.setZero();
  core::seat_index_t cp = Rules::get_current_player(*view.cur_pos);
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      core::seat_index_t p = get_player_at(*view.final_pos, row, col);
      int x = (p == -1) ? 2 : ((p == cp) ? 0 : 1);
      tensor(x, row, col) = 1;
    }
  }

  return tensor;
}

inline int Game::get_count(const State& state, core::seat_index_t seat) {
  if (seat == state.cur_player) {
    return std::popcount(state.cur_player_mask);
  } else {
    return std::popcount(state.opponent_mask);
  }
}

inline core::seat_index_t Game::get_player_at(const State& state, int row, int col) {
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
  mask_t flipped;

  flipped = (((P << dir) | (P >> dir)) & mask);
  flipped |= (((flipped << dir) | (flipped >> dir)) & mask);
  flipped |= (((flipped << dir) | (flipped >> dir)) & mask);
  flipped |= (((flipped << dir) | (flipped >> dir)) & mask);
  flipped |= (((flipped << dir) | (flipped >> dir)) & mask);
  flipped |= (((flipped << dir) | (flipped >> dir)) & mask);
  return (flipped << dir) | (flipped >> dir);

#endif
}

}  // namespace othello
