#include "games/othello/aux_features/StableDiscs.hpp"

#include <iostream>

namespace othello {

inline mask_t compute_stable_discs(mask_t cur_player_mask, mask_t opponent_mask) {

  mask_t stable_curr = edge_stable_discs(cur_player_mask);
  mask_t stable_oppo = edge_stable_discs(opponent_mask);

  full_edge_stable_discs(cur_player_mask, opponent_mask, stable_curr, stable_oppo);

  stable_curr |= stable_discs_protected_by_axes(cur_player_mask, stable_curr);
  stable_oppo |= stable_discs_protected_by_axes(opponent_mask, stable_oppo);
  mask_t stable = stable_curr | stable_oppo;

  return stable;
}

inline void full_edge_stable_discs(mask_t cur_player_mask, mask_t opponent_mask,
                                   mask_t& stable_curr, mask_t& stable_oppo) {
  auto edge_full_mask = [&](mask_t edge_mask) {
    mask_t filled = (cur_player_mask | opponent_mask) & edge_mask;
    mask_t fullMask = -(filled == edge_mask);
    stable_curr |= fullMask & edge_mask & cur_player_mask;
    stable_oppo |= fullMask & edge_mask & opponent_mask;
  };

  edge_full_mask(kRank1Mask);
  edge_full_mask(kRank8Mask);
  edge_full_mask(kFileAMask);
  edge_full_mask(kFileHMask);
}

inline mask_t edge_stable_discs(mask_t mask) {
  auto grow = [&](auto shift) {
    return [&, shift](mask_t start_bit, mask_t line_mask) {
      mask_t run = 0;
      mask_t m = start_bit;
      while (m & line_mask & mask) {
        run |= m;
        m = shift(m);
      }
      return run;
    };
  };

  auto grow_east = grow(step_east);
  auto grow_west = grow(step_west);
  auto grow_north = grow(step_north);
  auto grow_south = grow(step_south);

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

inline mask_t stable_discs_protected_by_axes(mask_t mask, mask_t stable) {
  std::cout << "Initial stable: ";
  std::cout << std::hex << stable << std::dec << "\n";
  auto ray_grow = [&](auto shift) {
    mask_t out = 0;
    mask_t frontier = stable;
    while (frontier) {
      frontier = shift(frontier) & mask & ~out;
      out |= frontier;
    }
    return out;
  };

  while (true) {
    const mask_t e  = ray_grow(step_east);
    const mask_t w  = ray_grow(step_west);
    const mask_t n  = ray_grow(step_north);
    const mask_t s  = ray_grow(step_south);
    const mask_t ne = ray_grow(step_northeast);
    const mask_t sw = ray_grow(step_southwest);
    const mask_t nw = ray_grow(step_northwest);
    const mask_t se = ray_grow(step_southeast);

    mask_t horiz, vert, diag1, diag2;

    horiz = e | w;
    vert = n | s;
    diag1 = ne | sw;
    diag2 = nw | se;

    const mask_t candidates = horiz & vert & diag1 & diag2;
    const mask_t newS = candidates & ~stable;
    if (!newS) break;
    stable |= newS;
  }
  std::cout << "Final stable: ";
  std::cout << std::hex << stable << std::dec << "\n";
  return stable;
}

}  // namespace othello
