#include "games/tictactoe/Game.hpp"
#include "games/tictactoe/PolicyEncoding.hpp"
#include "games/tictactoe/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

/*
 * Tests tictactoe symmetry classes.
 */

using Game = tictactoe::Game;
using Symmetries = tictactoe::Symmetries;
using State = Game::State;
using Move = Game::Move;
using MoveSet = Game::MoveSet;
using PolicyEncoding = tictactoe::PolicyEncoding;
using PolicyTensor = PolicyEncoding::Tensor;
using IO = Game::IO;
using Rules = Game::Rules;

template <typename... Ts>
State make_state(Ts... moves) {
  State state;
  Rules::init_state(state);

  for (Move move : {moves...}) {
    Rules::apply(state, move);
  }
  return state;
}

State make_init_state() { return make_state(7, 2); }

PolicyTensor make_policy(Move move1, Move move2) {
  PolicyTensor tensor;
  tensor.setZero();
  tensor(move1) = 1;
  tensor(move2) = 1;
  return tensor;
}

const std::string init_state_repr =
  "0 1 2  | | |O|\n"
  "3 4 5  | | | |\n"
  "6 7 8  | |X| |\n";

std::string get_repr(const State& state) {
  std::ostringstream ss;
  IO::print_state(ss, state);

  std::string s = ss.str();
  std::vector<std::string> lines;
  std::istringstream iss(s);
  std::string line;
  while (std::getline(iss, line)) {
    lines.push_back(line);
  }

  std::string repr;
  for (int i = 0; i < 3; ++i) {
    repr += lines[i];
    repr += '\n';
  }

  return repr;
}

TEST(Analyze, FromInitState) {
  State state;
  Rules::init_state(state);

  auto valid_moves = Rules::analyze(state).valid_moves();
  EXPECT_TRUE(valid_moves.all());
}

TEST(Symmetry, identity) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kIdentity;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr = init_state_repr;

  EXPECT_EQ(repr, expected_repr);
  Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(0, 1);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, rot90_clockwise) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kRot90;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "0 1 2  | | | |\n"
    "3 4 5  |X| | |\n"
    "6 7 8  | | |O|\n";

  EXPECT_EQ(repr, expected_repr);
  Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(2, 5);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, rot180) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kRot180;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "0 1 2  | |X| |\n"
    "3 4 5  | | | |\n"
    "6 7 8  |O| | |\n";

  EXPECT_EQ(repr, expected_repr);
  Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(7, 8);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, rot270_clockwise) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kRot270;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "0 1 2  |O| | |\n"
    "3 4 5  | | |X|\n"
    "6 7 8  | | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(3, 6);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, flip_vertical) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kFlipVertical;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "0 1 2  | |X| |\n"
    "3 4 5  | | | |\n"
    "6 7 8  | | |O|\n";

  EXPECT_EQ(repr, expected_repr);
  Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(6, 7);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, mirror_horizontal) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kMirrorHorizontal;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "0 1 2  |O| | |\n"
    "3 4 5  | | | |\n"
    "6 7 8  | |X| |\n";

  EXPECT_EQ(repr, expected_repr);
  Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(1, 2);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, flip_main_diag) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kFlipMainDiag;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "0 1 2  | | | |\n"
    "3 4 5  | | |X|\n"
    "6 7 8  |O| | |\n";

  EXPECT_EQ(repr, expected_repr);
  Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(0, 3);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, flip_anti_diag) {
  State state = make_init_state();

  group::element_t sym = groups::D4::kFlipAntiDiag;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "0 1 2  | | |O|\n"
    "3 4 5  |X| | |\n"
    "6 7 8  | | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(5, 8);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, canonicalization) {
  State state = make_state(2, 1);

  std::string expected_repr =
    "0 1 2  | |O|X|\n"
    "3 4 5  | | | |\n"
    "6 7 8  | | | |\n";

  std::string repr = get_repr(state);

  EXPECT_EQ(repr, expected_repr);

  group::element_t e = Symmetries::get_canonical_symmetry(state);
  EXPECT_EQ(e, groups::D4::kMirrorHorizontal);

  Symmetries::apply(state, e);

  expected_repr =
    "0 1 2  |X|O| |\n"
    "3 4 5  | | | |\n"
    "6 7 8  | | | |\n";

  repr = get_repr(state);

  EXPECT_EQ(repr, expected_repr);
}

TEST(Move, RoundTrip) {
  State state;
  Rules::init_state(state);

  // Index i serializes as the string "i"
  for (int i = 0; i < tictactoe::kNumCells; ++i) {
    tictactoe::Move m(i);
    std::string s = m.to_str();
    EXPECT_EQ(s, std::to_string(i)) << "index=" << i;
    tictactoe::Move back = tictactoe::Move::from_str(state, s);
    EXPECT_EQ(back, m) << "round-trip failed for index=" << i;
  }
}

// Board cell indices:
//   0 1 2
//   3 4 5
//   6 7 8
// Player X (0) moves first; player O (1) moves second.
// Sequence alternates X, O, X, O, ...

TEST(Rules, WinRow) {
  // X fills top row (0,1,2): X@0, O@3, X@1, O@4, X@2 => X wins
  State state = make_state(0, 3, 1, 4, 2);
  auto result = Rules::analyze(state);
  EXPECT_TRUE(result.is_terminal());
  EXPECT_EQ(result.outcome()(tictactoe::kX), 1.0f);
  EXPECT_EQ(result.outcome()(tictactoe::kO), 0.0f);
}

TEST(Rules, WinColumn) {
  // X fills left column (0,3,6): X@0, O@1, X@3, O@2, X@6 => X wins
  State state = make_state(0, 1, 3, 2, 6);
  auto result = Rules::analyze(state);
  EXPECT_TRUE(result.is_terminal());
  EXPECT_EQ(result.outcome()(tictactoe::kX), 1.0f);
  EXPECT_EQ(result.outcome()(tictactoe::kO), 0.0f);
}

TEST(Rules, WinDiagonal) {
  // X fills main diagonal (0,4,8): X@0, O@1, X@4, O@2, X@8 => X wins
  State state = make_state(0, 1, 4, 2, 8);
  auto result = Rules::analyze(state);
  EXPECT_TRUE(result.is_terminal());
  EXPECT_EQ(result.outcome()(tictactoe::kX), 1.0f);
  EXPECT_EQ(result.outcome()(tictactoe::kO), 0.0f);
}

TEST(Rules, Draw) {
  // Fill all 9 cells with no winner:
  //   X O X
  //   O X X
  //   O X O
  // X: 0,2,4,5,7  O: 1,3,6,8  (X moves 1st, so 5 X and 4 O)
  State state = make_state(0, 1, 2, 3, 4, 6, 5, 8, 7);
  auto result = Rules::analyze(state);
  EXPECT_TRUE(result.is_terminal());
  // WinLossDrawResults tensor is (W, L, D): draw is slot 2
  EXPECT_EQ(result.outcome()(2), 1.0f);  // D slot
  EXPECT_EQ(result.outcome()(0), 0.0f);  // no X win
  EXPECT_EQ(result.outcome()(1), 0.0f);  // no O win
}

TEST(Rules, MoveCount) {
  // After k non-terminal moves, valid move count should be 9 - k
  State state;
  Rules::init_state(state);
  EXPECT_EQ(Rules::analyze(state).valid_moves().size(), 9);

  Rules::apply(state, Move(0));
  EXPECT_EQ(Rules::analyze(state).valid_moves().size(), 8);

  Rules::apply(state, Move(1));
  EXPECT_EQ(Rules::analyze(state).valid_moves().size(), 7);

  Rules::apply(state, Move(3));
  EXPECT_EQ(Rules::analyze(state).valid_moves().size(), 6);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
