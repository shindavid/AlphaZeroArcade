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

inline mask_t edge_stable_discs(mask_t mask) {
  auto grow = [&](auto shift) {
    return [&](mask_t start_bit, mask_t line_mask) {
      mask_t run = 0;
      mask_t m = start_bit;
      while (m & line_mask & mask) {
        run |= m;
        m = shift(m);
      }
      return run;
    };
  };

  auto grow_east = grow([&](mask_t m) { return m << 1; });
  auto grow_west = grow([&](mask_t m) { return m >> 1; });
  auto grow_north = grow([&](mask_t m) { return m >> 8; });
  auto grow_south = grow([&](mask_t m) { return m << 8; });

  mask_t stable = 0;

  stable |= grow_east(kA1Mask, kRank1Mask);
  stable |= grow_west(kH1Mask, kRank1Mask);

  stable |= grow_east(kA8Mask, kRank8Mask);
  stable |= grow_west(kH8Mask, kRank8Mask);

  stable |= grow_south(kA1Mask, kFileAMask);
  stable |= grow_north(kA8Mask, kFileAMask);

  stable |= grow_south(kH1Mask, kFileHMask);
  stable |= grow_north(kH8Mask, kFileHMask);

  return stable;
}

}  // namespace othello
