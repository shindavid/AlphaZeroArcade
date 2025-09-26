#include "games/othello/aux_features/StableDiscs.hpp"

#include <gtest/gtest.h>

namespace othello {

TEST(Corner, stable) {
  mask_t cur_player_mask = 0x0000000000000011ULL;
  mask_t opponent_mask = 0x0000000000000000ULL;
  mask_t stable = compute_stable_discs(cur_player_mask, opponent_mask);
  EXPECT_EQ(stable, 0x0000000000000001ULL);

  cur_player_mask = 0x8000000000000011ULL;
  opponent_mask = 0x0000000000000000ULL;
  stable = compute_stable_discs(cur_player_mask, opponent_mask);
  EXPECT_EQ(stable, 0x8000000000000001ULL);

  cur_player_mask = 0x8000000000000011ULL;
  opponent_mask = 0x7FFFFFFFFFFFFFEFULL;
  stable = compute_stable_discs(cur_player_mask, opponent_mask);
  EXPECT_EQ(stable, 0xFFFFFFFFFFFFFFFFULL);
}



} // namespace othello
