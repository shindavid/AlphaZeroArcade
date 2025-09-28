#pragma once

#include "games/othello/Constants.hpp"

namespace othello {

  mask_t compute_stable_discs(mask_t cur_player_mask, mask_t opponent_mask);

  inline mask_t corner_stable_discs(mask_t mask) { return kCornersMask & mask; };
  mask_t edge_stable_discs(mask_t mask);

  inline constexpr auto step_east = [](mask_t m) { return m << 1; };
  inline constexpr auto step_west = [](mask_t m) { return m >> 1; };
  inline constexpr auto step_north = [](mask_t m) { return m >> 8; };
  inline constexpr auto step_south = [](mask_t m) { return m << 8; };
  inline constexpr auto step_northeast = [](mask_t m) { return m >> 7; };
  inline constexpr auto step_northwest = [](mask_t m) { return m >> 9; };
  inline constexpr auto step_southeast = [](mask_t m) { return m << 9; };
  inline constexpr auto step_southwest = [](mask_t m) { return m << 7; };

} // namespace othello

#include "inline/games/othello/aux_features/StableDiscs.inl"
