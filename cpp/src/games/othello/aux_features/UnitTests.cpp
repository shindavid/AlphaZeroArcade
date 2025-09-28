#include "games/othello/aux_features/StableDiscs.hpp"

#include <gtest/gtest.h>

namespace othello {

TEST(StableDiscs, corners) {
/*
   A B C D E F G H            A B C D E F G H
 1|O| | | |O| | | |         1|*| | | | | | | |
 2| | | | | | | | |         2| | | | | | | | |
 3| | | | | | | | |         3| | | | | | | | |
 4| | | | | | | | |   -->   4| | | | | | | | |
 5| | | | | | | | |         5| | | | | | | | |
 6| | | | | | | | |         6| | | | | | | | |
 7| | | | | | | | |         7| | | | | | | | |
 8| | | | | | | | |         8| | | | | | | | |

  Stable discs are A1
*/
  mask_t cur_player_mask = 0x0000000000000011ULL;
  mask_t opponent_mask = 0x0000000000000000ULL;
  mask_t stable = compute_stable_discs(cur_player_mask, opponent_mask);
  EXPECT_EQ(stable, 0x0000000000000001ULL);

/*
   A B C D E F G H            A B C D E F G H
 1|O| | | |O| | | |         1|*| | | | | | | |
 2| | | | | | | | |         2| | | | | | | | |
 3| | | | | | | | |         3| | | | | | | | |
 4| | | | | | | | |   -->   4| | | | | | | | |
 5| | | | | | | | |         5| | | | | | | | |
 6| | | | | | | | |         6| | | | | | | | |
 7| | | | | | | | |         7| | | | | | | | |
 8| | | | | | | |O|         8| | | | | | | |*|

  Stable discs are A1, H8
*/
  cur_player_mask = 0x8000000000000011ULL;
  opponent_mask = 0x0000000000000000ULL;
  stable = compute_stable_discs(cur_player_mask, opponent_mask);
  EXPECT_EQ(stable, 0x8000000000000001ULL);

/*
   A B C D E F G H            A B C D E F G H
 1|O| | | |O| | |X|         1|*| | | | | | |*|
 2| | | | | | | | |         2| | | | | | | | |
 3| | | | | | | | |         3| | | | | | | | |
 4| | | | | | | | |   -->   4| | | | | | | | |
 5| | | | | | | | |         5| | | | | | | | |
 6| | | | | | | | |         6| | | | | | | | |
 7| | | | | | | | |         7| | | | | | | | |
 8|X| | | | | | |O|         8|*| | | | | | |*|

  Stable discs are A1, A8, H1, H8
*/
  cur_player_mask = 0x8000000000000011ULL;
  opponent_mask = 0x0100000000000080ULL;
  stable = compute_stable_discs(cur_player_mask, opponent_mask);
  EXPECT_EQ(stable, 0x8100000000000081ULL);
}

TEST(StableDiscs, edges) {
/*
   A B C D E F G H            A B C D E F G H
 1|O|O| | | | | | |         1|*|*| | | | | | |
 2| | | | | | | | |         2| | | | | | | | |
 3| | | | | | | | |         3| | | | | | | | |
 4| | | | | | | | |   -->   4| | | | | | | | |
 5| | | | | | | | |         5| | | | | | | | |
 6| | | | | | | | |         6| | | | | | | | |
 7| | | | | | | | |         7| | | | | | | | |
 8| | | | | | | | |         8| | | | | | | | |

  Stable discs are A1, B1
*/
  mask_t cur_player_mask = 0x0000000000000003ULL;
  mask_t opponent_mask = 0x0000000000000000ULL;
  mask_t stable = compute_stable_discs(cur_player_mask, opponent_mask);
  EXPECT_EQ(stable, 0x0000000000000003ULL);

/*
   A B C D E F G H            A B C D E F G H
 1|O|O|O|O|O|O|O|O|         1|*|*|*|*|*|*|*|*|
 2| | | | | | | | |         2| | | | | | | | |
 3| | | | | | | | |         3| | | | | | | | |
 4| | | | | | | | |   -->   4| | | | | | | | |
 5| | | | | | | | |         5| | | | | | | | |
 6| | | | | | | | |         6| | | | | | | | |
 7| | | | | | | | |         7| | | | | | | | |
 8| | | | | | | | |         8| | | | | | | | |

  Stable discs are A1, B1, C1, D1, E1, F1, G1, H1
*/
  cur_player_mask = 0x00000000000000FFULL;
  opponent_mask = 0x0000000000000000ULL;
  stable = compute_stable_discs(cur_player_mask, opponent_mask);
  EXPECT_EQ(stable, 0x00000000000000FFULL);

/*
   A B C D E F G H            A B C D E F G H
 1| | | | | | | | |         1| | | | | | | | |
 2| | | | | | | | |         2| | | | | | | | |
 3| | | | | | | | |         3| | | | | | | | |
 4| | | | | | | | |   -->   4| | | | | | | | |
 5| | | | | | | | |         5| | | | | | | | |
 6| | | | | | | | |         6| | | | | | | | |
 7| | | | | | | |O|         7| | | | | | | |*|
 8| | | | | | | |O|         8| | | | | | | |*|

  Stable discs are H7, H8
*/
  cur_player_mask = 0x8080000000000000ULL;
  opponent_mask = 0x0000000000000000ULL;
  stable = compute_stable_discs(cur_player_mask, opponent_mask);
  EXPECT_EQ(stable, 0x8080000000000000ULL);


/*
   A B C D E F G H            A B C D E F G H
 1| | | | | |X|X|X|         1| | | | | |*|*|*|
 2| | | | | | | |O|         2| | | | | | | | |
 3| | | | | | | |O|         3| | | | | | | | |
 4|O| | | | | | | |   -->   4|*| | | | | | | |
 5|O| | | | | | | |         5|*| | | | | | | |
 6|O| | | | | | | |         6|*| | | | | | | |
 7|O| | | | | | | |         7|*| | | | | | | |
 8|O| | | | | | | |         8|*| | | | | | | |

  Stable discs are A4, A5, A6, A7, A8, F1, G1, H1
*/
  cur_player_mask = 0x0101010101808000ULL;
  opponent_mask = 0x00000000000000E0ULL;
  stable = compute_stable_discs(cur_player_mask, opponent_mask);
  EXPECT_EQ(stable, 0x01010101010000E0ULL);
}

} // namespace othello
