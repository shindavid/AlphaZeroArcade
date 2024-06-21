#include <games/othello/Game.hpp>

#include <algorithm>
#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <util/AnsiCodes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

inline std::size_t std::hash<othello::GameState>::operator()(
    const othello::GameState& state) const {
  return state.hash();
}

namespace othello {

inline GameState::SymmetryIndexSet GameState::get_symmetries() const {
  SymmetryIndexSet set;
  set.set();
  return set;
}

inline int GameState::get_count(core::seat_index_t seat) const {
  if (seat == data_.cur_player) {
    return std::popcount(data_.cur_player_mask);
  } else {
    return std::popcount(data_.opponent_mask);
  }
}

inline core::seat_index_t GameState::get_player_at(int row, int col) const {
  int cp = get_current_player();
  int index = row * kBoardDimension + col;
  bool occupied_by_cur_player = (mask_t(1) << index) & data_.cur_player_mask;
  bool occupied_by_opponent = (mask_t(1) << index) & data_.opponent_mask;
  return occupied_by_opponent ? (1 - cp) : (occupied_by_cur_player ? cp : -1);
}

inline std::size_t GameState::hash() const { return util::tuple_hash(to_tuple()); }

// copied from edax-reversi repo
inline mask_t GameState::get_moves(mask_t P, mask_t O) {
  mask_t mask = O & 0x7E7E7E7E7E7E7E7Eull;

  return (get_some_moves(P, mask, 1)    // horizontal
          | get_some_moves(P, O, 8)     // vertical
          | get_some_moves(P, mask, 7)  // diagonals
          | get_some_moves(P, mask, 9)) &
         ~(P | O);  // mask with empties
}

// copied from edax-reversi repo
inline mask_t GameState::get_some_moves(mask_t P, mask_t mask, int dir) {
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
