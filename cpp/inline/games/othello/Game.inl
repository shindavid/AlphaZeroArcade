#include "games/othello/Game.hpp"

#include "core/DefaultCanonicalizer.hpp"
#include "util/BitMapUtil.hpp"

#include <boost/lexical_cast.hpp>

#include <bit>

namespace othello {

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
    case D4::kIdentity:
      return;
    case D4::kRot90:
      return rot90_clockwise(s.core.cur_player_mask, s.core.opponent_mask);
    case D4::kRot180:
      return rot180(s.core.cur_player_mask, s.core.opponent_mask);
    case D4::kRot270:
      return rot270_clockwise(s.core.cur_player_mask, s.core.opponent_mask);
    case D4::kFlipVertical:
      return flip_vertical(s.core.cur_player_mask, s.core.opponent_mask);
    case D4::kFlipMainDiag:
      return flip_main_diag(s.core.cur_player_mask, s.core.opponent_mask);
    case D4::kMirrorHorizontal:
      return mirror_horizontal(s.core.cur_player_mask, s.core.opponent_mask);
    case D4::kFlipAntiDiag:
      return flip_anti_diag(s.core.cur_player_mask, s.core.opponent_mask);
    default:
      throw util::Exception("Unknown group element: {}", sym);
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
    default:
      throw util::Exception("Unknown group element: {}", sym);
  }
}

inline void Game::Symmetries::apply(core::action_t& action, group::element_t sym,
                                    core::action_mode_t) {
  using namespace bitmap_util;
  using D4 = groups::D4;

  if (action == kPass || sym == D4::kIdentity) return;

  mask_t mask = 1ULL << action;

  switch (sym) {
    case D4::kRot90:
      rot90_clockwise(mask);
      break;
    case D4::kRot180:
      rot180(mask);
      break;
    case D4::kRot270:
      rot270_clockwise(mask);
      break;
    case D4::kFlipVertical:
      flip_vertical(mask);
      break;
    case D4::kFlipMainDiag:
      flip_main_diag(mask);
      break;
    case D4::kMirrorHorizontal:
      mirror_horizontal(mask);
      break;
    case D4::kFlipAntiDiag:
      flip_anti_diag(mask);
      break;
    default:
      throw util::Exception("Unknown group element: {}", sym);
  }

  action = std::countr_zero(mask);
}

inline group::element_t Game::Symmetries::get_canonical_symmetry(const State& state) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<Game>;
  return DefaultCanonicalizer::get(state);
}

inline void Game::Rules::init_state(State& state) {
  state.core.opponent_mask = kStartingWhiteMask;
  state.core.cur_player_mask = kStartingBlackMask;
  state.core.cur_player = kStartingColor;
  state.core.pass_count = 0;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.core.cur_player;
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
