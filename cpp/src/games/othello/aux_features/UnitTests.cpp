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

TEST(StableDiscs, interior_one_side_protected) {

/*
   A B C D E F G H            A B C D E F G H
 1|O|O|O| | |X|X|X|         1|*|*|*| | |*|*|*|
 2|O|O| | | | |O|O|         2|*|*| | | | | | |
 3| | | | | | | |O|         3| | | | | | | | |
 4| | | | | | | | |   -->   4| | | | | | | | |
 5| | | | | | | | |         5| | | | | | | | |
 6| | | | | | | | |         6| | | | | | | | |
 7| | | | | | | | |         7| | | | | | | | |
 8| | | | | | | | |         8| | | | | | | | |

  Stable discs are A1, A2, B1, B2, C1, F1, G1, H1
*/

mask_t cur_player_mask = (1ULL << kF1) | (1ULL << kG1) | (1ULL << kH1);
mask_t opponent_mask = (1ULL << kA1) | (1ULL << kA2) | (1ULL << kB1) | (1ULL << kB2) |
                     (1ULL << kC1) | (1ULL << kG2) | (1ULL << kH2) | (1ULL << kH3);
mask_t stable = compute_stable_discs(cur_player_mask, opponent_mask);
mask_t expected_stable = (1ULL << kA1) | (1ULL << kA2) | (1ULL << kB1) | (1ULL << kB2) |
                         (1ULL << kC1) | (1ULL << kF1) | (1ULL << kG1) | (1ULL << kH1);
EXPECT_EQ(stable, expected_stable);
}

TEST(StableDiscs, full_rank1) {
/*
   A B C D E F G H            A B C D E F G H
 1|O|O|O|X|O|X|X|X|         1|*|*|*|*|*|*|*|*|
 2| | | | | | | | |         2| | | | | | | | |
 3| | | | | | | | |         3| | | | | | | | |
 4| | | | | | | | |   -->   4| | | | | | | | |
 5| | | | | | | | |         5| | | | | | | | |
 6| | | | | | | | |         6| | | | | | | | |
 7| | | | | | | | |         7| | | | | | | | |
 8| | | | | | | | |         8| | | | | | | | |

  Stable discs are A1, A2, A3, B1, B2, B3, C1, C2, C3, D1, E1, F1, G1, H1
*/
mask_t cur_player_mask = (1ULL << kD1) | (1ULL << kF1) | (1ULL << kG1) | (1ULL << kH1);
mask_t opponent_mask = (1ULL << kA1) | (1ULL << kB1) | (1ULL << kC1) | (1ULL << kE1);
mask_t stable = compute_stable_discs(cur_player_mask, opponent_mask);
mask_t expected_stable = (1ULL << kA1) | (1ULL << kB1) | (1ULL << kC1) | (1ULL << kD1) |
                         (1ULL << kE1) | (1ULL << kF1) | (1ULL << kG1) | (1ULL << kH1);
EXPECT_EQ(stable, expected_stable);
}

TEST(StableDiscs, full_rank8) {
/*
   A B C D E F G H            A B C D E F G H
 1| | | | | | | | |         1| | | | | | | | |
 2| | | | | | | | |         2| | | | | | | | |
 3| | | | | | | | |         3| | | | | | | | |
 4| | | | | | | | |   -->   4| | | | | | | | |
 5| | | | | | | | |         5| | | | | | | | |
 6| | | | | | | | |         6| | | | | | | | |
 7| | | | | | | | |         7| | | | | | | | |
 8|O|O|O|X|O|X|X|X|         8|*|*|*|*|*|*|*|*|

    Stable discs are A8, B8, C8, D8, E8, F8, G8, H8
*/
mask_t cur_player_mask = (1ULL << kD8) | (1ULL << kF8) | (1ULL << kG8) | (1ULL << kH8);
mask_t opponent_mask = (1ULL << kA8) | (1ULL << kB8) | (1ULL << kC8) | (1ULL << kE8);
mask_t stable = compute_stable_discs(cur_player_mask, opponent_mask);
mask_t expected_stable = (1ULL << kA8) | (1ULL << kB8) | (1ULL << kC8) | (1ULL << kD8) |
                         (1ULL << kE8) | (1ULL << kF8) | (1ULL << kG8) | (1ULL << kH8);
EXPECT_EQ(stable, expected_stable);
}

TEST(StableDiscs, full_fileA) {
/*
   A B C D E F G H            A B C D E F G H
 1|O| | | | | | | |         1|*| | | | | | | |
 2|X| | | | | | | |         2|*| | | | | | | |
 3|X| | | | | | | |         3|*| | | | | | | |
 4|O| | | | | | | |   -->   4|*| | | | | | | |
 5|X| | | | | | | |         5|*| | | | | | | |
 6|O| | | | | | | |         6|*| | | | | | | |
 7|X| | | | | | | |         7|*| | | | | | | |
 8|O| | | | | | | |         8|*| | | | | | | |

    Stable discs are A1, A2, A3, A4, A5, A6, A7, A8
*/

mask_t cur_player_mask = (1ULL << kA2) | (1ULL << kA3) | (1ULL << kA5) | (1ULL << kA7);
mask_t opponent_mask = (1ULL << kA1) | (1ULL << kA4) | (1ULL << kA6) | (1ULL << kA8);
mask_t stable = compute_stable_discs(cur_player_mask, opponent_mask);
mask_t expected_stable = (1ULL << kA1) | (1ULL << kA2) | (1ULL << kA3) | (1ULL << kA4) |
                         (1ULL << kA5) | (1ULL << kA6) | (1ULL << kA7) | (1ULL << kA8);
EXPECT_EQ(stable, expected_stable);
}


TEST(StableDiscs, full_axes) {
/*
   A B C D E F G H            A B C D E F G H
 1|O|O|O| | | | | |         1|*|*|*| | | | | |
 2|X|X|O|O|X|X|O|X|         2| |*| | | | | | |
 3|X|O|X| | | | | |         3| | | | | | | | |
 4| |O| |O| | | | |   -->   4| | | | | | | | |
 5| |O| | |O| | | |         5| | | | | | | | |
 6| |O| | | |X| | |         6| | | | | | | | |
 7| |O| | | | |X| |         7| | | | | | | | |
 8| |O| | | | | |X|         8| | | | | | | |*|

    Stable discs are A1, B1, C1, B2
*/

mask_t cur_player_mask = (1ULL << kA2) | (1ULL << kA3) | (1ULL) << kB2 | (1ULL << kC3) |
                         (1ULL << kE2) | (1ULL << kF2) | (1ULL << kF6) | (1ULL << kG7) |
                         (1ULL << kH2) | (1ULL << kH8);

mask_t opponent_mask = (1ULL << kA1) | (1ULL << kB1) | (1ULL << kB3) | (1ULL << kB4) |
                       (1ULL << kB5) | (1ULL << kB6) | (1ULL << kB7) | (1ULL << kB8) |
                       (1ULL << kC1) | (1ULL << kC2) | (1ULL << kD2) | (1ULL << kD4) |
                       (1ULL << kE5) | (1ULL << kG2);

mask_t stable = compute_stable_discs(cur_player_mask, opponent_mask);
mask_t expected_stable =
  (1ULL << kA1) | (1ULL << kB1) | (1ULL << kC1) | (1ULL << kB2) | (1ULL << kH8);
EXPECT_EQ(stable, expected_stable);
}

TEST(StableDiscs, edge_with_empty_space) {
/*
   A B C D E F G H            A B C D E F G H
 1|O|O|X|O| |O|X|O|         1|*|*|*| | | |*|*|
 2| | | | | | | | |         2| | | | | | | | |
 3| | | | | | | | |         3| | | | | | | | |
 4| | | | | | | | |   -->   4| | | | | | | | |
 5| | | | | | | | |         5| | | | | | | | |
 6| | | | | | | | |         6| | | | | | | | |
 7| | | | | | | | |         7| | | | | | | | |
 8| | | | | | | | |         8| | | | | | | | |

    Stable discs are A1, B1, C1, G1, H1
*/

mask_t cur_player_mask = (1ULL << kC1) | (1ULL << kG1);

mask_t opponent_mask =
  (1ULL << kA1) | (1ULL << kB1) | (1ULL << kD1) | (1ULL << kF1) | (1ULL << kH1);

mask_t stable = compute_stable_discs(cur_player_mask, opponent_mask);
mask_t expected_stable =
  (1ULL << kA1) | (1ULL << kB1) | (1ULL << kC1) | (1ULL << kG1) | (1ULL << kH1);
EXPECT_EQ(stable, expected_stable);
}

} // namespace othello
