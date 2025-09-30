#pragma once

#include "games/othello/Constants.hpp"

namespace othello {

  mask_t compute_stable_discs(mask_t cur_player_mask, mask_t opponent_mask);


  void full_edge_stable_discs(mask_t cur_player_mask, mask_t opponent_mask, mask_t& stable_curr,
                              mask_t& stable_oppo);
  void full_axes_stable_discs(mask_t cur_player_mask, mask_t opponent_mask, mask_t& stable_curr,
                              mask_t& stable_oppo);
  mask_t edge_stable_discs(mask_t mask);
  mask_t extend_stable_frontier(mask_t mask, mask_t stable);

  static inline uint64_t bit64(int i);
  static inline uint64_t unpack_file_middle(uint8_t m, int file);
  static inline uint8_t edge_stable_lookup(uint8_t p8, uint8_t o8);
  static inline uint64_t get_stable_edge_scalar(uint64_t P, uint64_t O);

  inline mask_t step_east(mask_t m) { return (m & ~kFileHMask) << 1; }
  inline mask_t step_west(mask_t m) { return (m & ~kFileAMask) >> 1; }
  inline mask_t step_north(mask_t m) { return m >> 8; };
  inline mask_t step_south(mask_t m) { return m << 8; };
  inline mask_t step_northeast(mask_t m) { return (m & ~kFileHMask) >> 7; }
  inline mask_t step_northwest(mask_t m) { return (m & ~kFileAMask) >> 9; }
  inline mask_t step_southeast(mask_t m) { return (m & ~kFileHMask) << 9; }
  inline mask_t step_southwest(mask_t m) { return (m & ~kFileAMask) << 7; }

} // namespace othello

#include "inline/games/othello/aux_features/StableDiscs.inl"
