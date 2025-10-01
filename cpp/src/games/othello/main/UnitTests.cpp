#include "core/tests/Common.hpp"
#include "games/othello/aux_features/StableDiscs.hpp"
#include "games/othello/Constants.hpp"
#include "games/othello/Game.hpp"
#include "util/EigenUtil.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

/*
 * Tests othello symmetry classes.
 */

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using Game = othello::Game;
using State = Game::State;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

State make_init_state() {
  State state;
  Rules::init_state(state);
  Rules::apply(state, othello::kD3);
  return state;
}

PolicyTensor make_policy(int move) {
  PolicyTensor tensor;
  tensor.setZero();
  tensor(move) = 1;
  return tensor;
}

const std::string init_state_repr =
  "   A B C D E F G H\n"
  " 1| | | | | | | | |\n"
  " 2| | | | | | | | |\n"
  " 3| | |.|*|.| | | |\n"
  " 4| | | |*|*| | | |\n"
  " 5| | |.|*|0| | | |\n"
  " 6| | | | | | | | |\n"
  " 7| | | | | | | | |\n"
  " 8| | | | | | | | |\n";

std::string get_repr(const State& state) {
  std::ostringstream ss;
  IO::print_state(ss, state);

  std::string s = ss.str();

  // only use the first 9 lines, we don't care about score part

  std::vector<std::string> lines;
  std::istringstream iss(s);
  std::string line;
  while (std::getline(iss, line)) {
    lines.push_back(line);
  }

  std::string repr;
  for (int i = 0; i < 9; ++i) {
    repr += lines[i];
    repr += '\n';
  }

  return repr;
}

TEST(Symmetry, identity) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kIdentity;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr = init_state_repr;

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kA3);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, rot90_clockwise) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kRot90;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "   A B C D E F G H\n"
    " 1| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 3| | | |.| |.| | |\n"
    " 4| | | |*|*|*| | |\n"
    " 5| | | |0|*|.| | |\n"
    " 6| | | | | | | | |\n"
    " 7| | | | | | | | |\n"
    " 8| | | | | | | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kF1);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, rot180) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kRot180;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "   A B C D E F G H\n"
    " 1| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 4| | | |0|*|.| | |\n"
    " 5| | | |*|*| | | |\n"
    " 6| | | |.|*|.| | |\n"
    " 7| | | | | | | | |\n"
    " 8| | | | | | | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kH6);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, rot270_clockwise) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kRot270;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "   A B C D E F G H\n"
    " 1| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 4| | |.|*|0| | | |\n"
    " 5| | |*|*|*| | | |\n"
    " 6| | |.| |.| | | |\n"
    " 7| | | | | | | | |\n"
    " 8| | | | | | | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kC8);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, flip_vertical) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kFlipVertical;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "   A B C D E F G H\n"
    " 1| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 4| | |.|*|0| | | |\n"
    " 5| | | |*|*| | | |\n"
    " 6| | |.|*|.| | | |\n"
    " 7| | | | | | | | |\n"
    " 8| | | | | | | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kA6);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, mirror_horizontal) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kMirrorHorizontal;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "   A B C D E F G H\n"
    " 1| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 3| | | |.|*|.| | |\n"
    " 4| | | |*|*| | | |\n"
    " 5| | | |0|*|.| | |\n"
    " 6| | | | | | | | |\n"
    " 7| | | | | | | | |\n"
    " 8| | | | | | | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kH3);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, flip_main_diag) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kFlipMainDiag;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "   A B C D E F G H\n"
    " 1| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 3| | |.| |.| | | |\n"
    " 4| | |*|*|*| | | |\n"
    " 5| | |.|*|0| | | |\n"
    " 6| | | | | | | | |\n"
    " 7| | | | | | | | |\n"
    " 8| | | | | | | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kC1);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, flip_anti_diag) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kFlipAntiDiag;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "   A B C D E F G H\n"
    " 1| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 3| | | | | | | | |\n"
    " 4| | | |0|*|.| | |\n"
    " 5| | | |*|*|*| | |\n"
    " 6| | | |.| |.| | |\n"
    " 7| | | | | | | | |\n"
    " 8| | | | | | | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kF8);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, action_transforms) { core::tests::Common<Game>::gtest_action_transforms(); }

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

}  // namespace othello

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
