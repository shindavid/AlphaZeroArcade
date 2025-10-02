#pragma once

#include "games/othello/Constants.hpp"

namespace othello {

inline mask_t compute_stable_discs(mask_t cur_player_mask, mask_t opponent_mask,
                                   const mask_t& stable);

inline mask_t get_stable_edge(mask_t P, mask_t O);
void full_axes_stable_discs(mask_t cur_player_mask, mask_t opponent_mask, mask_t& stable_curr,
                            mask_t& stable_oppo);
mask_t extend_stable_frontier(mask_t mask, mask_t stable);


inline line_mask_t edge_stable_lookup(line_mask_t p8, line_mask_t o8);
inline line_mask_t pack_file(mask_t bb, int file);
inline mask_t unpack_file_middle(line_mask_t m, int file);

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
