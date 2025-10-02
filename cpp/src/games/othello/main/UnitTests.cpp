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
using mask_t = othello::mask_t;

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

struct Masks {
  mask_t cur_player = 0;
  mask_t opponent = 0;
  mask_t stable = 0;
};

constexpr int index(int row, int col) { return row * 8 + col; }

inline Masks parse_board(const std::string& ascii_board) {
  Masks m;
  std::istringstream iss(ascii_board);
  std::string line;
  int row = 0;

  std::getline(iss, line);  // skip header line "  A B C D E F G H"
  while (std::getline(iss, line)) {
    int row_idx = line[0] - '1';  // '1' -> row 0

    int col = 0;
    for (size_t i = 2; i < line.size() && col < 8; i += 2) {
      char ch = line[i];
      int bit = index(row_idx, col);
      mask_t mask = 1ULL << bit;

      switch (ch) {
        case 'X':
          m.cur_player |= mask;
          break;
        case 'O':
          m.opponent |= mask;
          break;
        case '*':
          m.stable |= mask;
          break;
        default:
          break;
      }
      ++col;
    }
    ++row;
  }
  return m;
}

TEST(StableDiscs, from_string) {
  const char* board =
    "  A B C D E F G H\n"
    "1|O|O|O| | | | | |\n"
    "2|X|X|O|O|X|X|O|X|\n"
    "3|X|O|X| | | | | |\n"
    "4| |O| |O| | | | |\n"
    "5| |O| | |O| | | |\n"
    "6| |O| | | |X| | |\n"
    "7| |O| | | | |X| |\n"
    "8| |O| | | | | |X|\n";

  const char* expected_stable =
    " A B C D E F G H\n"
    "1|*|*|*| | | | | |\n"
    "2| |*| | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | |*|\n";

  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

TEST(StableDiscs, corners) {
  const char* board =
    "  A B C D E F G H\n"
    "1|O| | |O| | | | |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | |X| | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | |X|\n";

  const char* expected_stable =
    " A B C D E F G H\n"
    "1|*| | | | | | | |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | |*|\n";

  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

TEST(StableDiscs, interior_one_side_protected) {
  const char* board =
    "  A B C D E F G H\n"
    "1|O|O|O| | |X|X|X|\n"
    "2|O|O| | | | |O|O|\n"
    "3| | | | | | | |O|\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  const char* expected_stable =
    " A B C D E F G H\n"
    "1|*|*|*| | |*|*|*|\n"
    "2|*|*| | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";
  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

TEST(StableDiscs, edge_with_one_empty_space) {
  const char* board =
    "  A B C D E F G H\n"
    "1|O|O|X|O| |O|X|O|\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  const char* expected_stable =
    " A B C D E F G H\n"
    "1|*|*|*| | | |*|*|\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

TEST(StableDiscs, edge_with_two_empty_spaces) {
  const char* board =
    "  A B C D E F G H\n"
    "1|O|O|X|O| | |X|O|\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  const char* expected_stable =
    " A B C D E F G H\n"
    "1|*|*| | | | | |*|\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

TEST(StableDiscs, file_with_one_empty_spaces) {
  const char* board =
    "  A B C D E F G H\n"
    "1|O|O|X|O| | |X|O|\n"
    "2|X| | | | | | | |\n"
    "3|X| | | | | | | |\n"
    "4|O| | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6|O| | | | | | | |\n"
    "7|O| | | | | | | |\n"
    "8|O| | | | | | | |\n";

  const char* expected_stable =
    " A B C D E F G H\n"
    "1|*|*| | | | | |*|\n"
    "2|*| | | | | | | |\n"
    "3|*| | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6|*| | | | | | | |\n"
    "7|*| | | | | | | |\n"
    "8|*| | | | | | | |\n";

  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

TEST(StableDiscs, unoccupied_corners) {
  const char* board =
    "  A B C D E F G H\n"
    "1| |O|X|O|X|X|X| |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  const char* expected_stable =
    " A B C D E F G H\n"
    "1| | |*|*| | | | |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

TEST(StableDiscs, unoccupied_corners_left_edge) {
  const char* board =
    "  A B C D E F G H\n"
    "1| | | | | | | | |\n"
    "2|X| | | | | | | |\n"
    "3|O| | | | | | | |\n"
    "4|X| | | | | | | |\n"
    "5|O| | | | | | | |\n"
    "6|X| | | | | | | |\n"
    "7|O| | | | | | | |\n"
    "8| | | | | | | | |\n";

  const char* expected_stable =
    " A B C D E F G H\n"
    "1| | | | | | | | |\n"
    "2| | | | | | | | |\n"
    "3|*| | | | | | | |\n"
    "4|*| | | | | | | |\n"
    "5|*| | | | | | | |\n"
    "6|*| | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

TEST(StableDiscs, unoccupied_corners_bottom) {
  const char* board =
    "  A B C D E F G H\n"
    "1| | | | | | | | |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | |X|O|X|O|X| |\n";

  const char* expected_stable =
    "  A B C D E F G H\n"
    "1| | | | | | | | |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | |*|*| | |\n";

  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

TEST(StableDiscs, unoccupied_corners_top) {
  const char* board =
    "  A B C D E F G H\n"
    "1| | |X|O|X|O|X| |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  const char* expected_stable =
    "  A B C D E F G H\n"
    "1| | | | |*|*| | |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

TEST(StableDiscs, unoccupied_corners_top2) {
  const char* board =
    "  A B C D E F G H\n"
    "1| |O|X|O|X|O| | |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  const char* expected_stable =
    "  A B C D E F G H\n"
    "1| | |*|*| | | | |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

TEST(StableDiscs, unoccupied_corners_top3) {
  const char* board =
    "  A B C D E F G H\n"
    "1| |X|O|X|O|X| | |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  const char* expected_stable =
    "  A B C D E F G H\n"
    "1| | |*|*| | | | |\n"
    "2| | | | | | | | |\n"
    "3| | | | | | | | |\n"
    "4| | | | | | | | |\n"
    "5| | | | | | | | |\n"
    "6| | | | | | | | |\n"
    "7| | | | | | | | |\n"
    "8| | | | | | | | |\n";

  Masks m = parse_board(board);
  Masks expected = parse_board(expected_stable);
  mask_t stable = othello::compute_stable_discs(m.cur_player, m.opponent);
  EXPECT_EQ(stable, expected.stable);
}

  int main(int argc, char** argv) { return launch_gtest(argc, argv); }
