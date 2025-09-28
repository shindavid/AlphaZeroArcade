#pragma once

#include "games/othello/Constants.hpp"

namespace othello {

  mask_t compute_stable_discs(mask_t cur_player_mask, mask_t opponent_mask);

  mask_t edge_stable_discs(mask_t mask);
  mask_t stable_discs_protected_by_axes(mask_t mask, mask_t stable, bool both_ends=false);

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
