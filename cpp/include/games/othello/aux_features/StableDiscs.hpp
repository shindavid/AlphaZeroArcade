#pragma once

#include "games/othello/Constants.hpp"

namespace othello {

  mask_t compute_stable_discs(mask_t cur_player_mask, mask_t opponent_mask);

  mask_t corner_stable_discs(mask_t cur_player_mask, mask_t opponent_mask);

} // namespace othello

#include "inline/games/othello/aux_features/StableDiscs.inl"
