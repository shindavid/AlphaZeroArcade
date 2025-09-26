#include "games/othello/aux_features/StableDiscs.hpp"

namespace othello {

inline mask_t compute_stable_discs(mask_t cur_player_mask, mask_t opponent_mask) {
  mask_t stable = 0;

  stable = stable | corner_stable_discs(cur_player_mask);
  stable = stable | corner_stable_discs(opponent_mask);

  stable = stable | edge_stable_discs(cur_player_mask);
  stable = stable | edge_stable_discs(opponent_mask);

  return stable;
}

inline mask_t corner_stable_discs(mask_t cur_player_mask) {
  mask_t A1 = 1ULL << kA1;
  mask_t A8 = 1ULL << kA8;
  mask_t H1 = 1ULL << kH1;
  mask_t H8 = 1ULL << kH8;

  mask_t stable = 0;
  if (cur_player_mask & A1) stable |= A1;
  if (cur_player_mask & A8) stable |= A8;
  if (cur_player_mask & H1) stable |= H1;
  if (cur_player_mask & H8) stable |= H8;
  return stable;
}

inline mask_t edge_stable_discs(mask_t cur_player_mask) {
  // Bitboard edge masks (A1 is LSB, H8 is MSB)
  constexpr mask_t RANK1 = 0x00000000000000FFULL;
  constexpr mask_t RANK8 = 0xFF00000000000000ULL;
  constexpr mask_t FILEA = 0x0101010101010101ULL;
  constexpr mask_t FILEH = 0x8080808080808080ULL;

  // Corner bits (consistent with the A1..H8 layout used in your tests)
  constexpr mask_t A1 = 0x0000000000000001ULL;
  constexpr mask_t H1 = 0x0000000000000080ULL;
  constexpr mask_t A8 = 0x0100000000000000ULL;
  constexpr mask_t H8 = 0x8000000000000000ULL;

  mask_t stable = 0;

  // Helper lambdas to grow a contiguous run of the current player's discs
  // from a corner along a single edge.
  auto grow_east = [&](mask_t start_bit, mask_t line_mask) {
    mask_t run = 0;
    mask_t m = start_bit;
    while ((m & line_mask) && (m & cur_player_mask)) {
      run |= m;
      m <<= 1; // move east along the rank
    }
    return run;
  };

  auto grow_west = [&](mask_t start_bit, mask_t line_mask) {
    mask_t run = 0;
    mask_t m = start_bit;
    while ((m & line_mask) && (m & cur_player_mask)) {
      run |= m;
      m >>= 1; // move west along the rank
    }
    return run;
  };

  auto grow_north = [&](mask_t start_bit, mask_t file_mask) {
    mask_t run = 0;
    mask_t m = start_bit;
    while ((m & file_mask) && (m & cur_player_mask)) {
      run |= m;
      m <<= 8; // move up the file
    }
    return run;
  };

  auto grow_south = [&](mask_t start_bit, mask_t file_mask) {
    mask_t run = 0;
    mask_t m = start_bit;
    while ((m & file_mask) && (m & cur_player_mask)) {
      run |= m;
      m >>= 8; // move down the file
    }
    return run;
  };

  // Bottom edge: from A1→H1 and from H1→A1 (anchored at corners if owned)
  if (cur_player_mask & A1) stable |= grow_east(A1, RANK1);
  if (cur_player_mask & H1) stable |= grow_west(H1, RANK1);

  // Top edge: from A8→H8 and from H8→A8
  if (cur_player_mask & A8) stable |= grow_east(A8, RANK8);
  if (cur_player_mask & H8) stable |= grow_west(H8, RANK8);

  // Left edge: from A1→A8 and from A8→A1
  if (cur_player_mask & A1) stable |= grow_north(A1, FILEA);
  if (cur_player_mask & A8) stable |= grow_south(A8, FILEA);

  // Right edge: from H1→H8 and from H8→H1
  if (cur_player_mask & H1) stable |= grow_north(H1, FILEH);
  if (cur_player_mask & H8) stable |= grow_south(H8, FILEH);

  return stable;
}

}  // namespace othello
