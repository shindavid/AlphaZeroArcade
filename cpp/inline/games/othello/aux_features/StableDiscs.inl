#include "games/othello/aux_features/StableDiscs.hpp"

namespace othello {

inline mask_t compute_stable_discs(mask_t cur_player_mask, mask_t opponent_mask) {
  mask_t stable = 0;
  stable = stable | corner_stable_discs(cur_player_mask, opponent_mask);
  return stable;
}

inline mask_t corner_stable_discs(mask_t cur_player_mask, mask_t opponent_mask) {
  mask_t A1 = 1ULL << kA1;
  mask_t A8 = 1ULL << kA8;
  mask_t H1 = 1ULL << kH1;
  mask_t H8 = 1ULL << kH8;

  mask_t stable = 0;
  if ((cur_player_mask | opponent_mask) & A1) stable |= A1;
  if ((cur_player_mask | opponent_mask) & A8) stable |= A8;
  if ((cur_player_mask | opponent_mask) & H1) stable |= H1;
  if ((cur_player_mask | opponent_mask) & H8) stable |= H8;
  return stable;
}

}  // namespace othello
