#include "games/connect4/Constants.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/PerfectOracle.hpp"
#include "games/connect4/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

/*
 * Connect4 tests.
 */

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using Game = c4::Game;
using Symmetries = c4::Symmetries;
using State = Game::State;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

State make_init_state() {
  State state;
  Rules::init_state(state);

  Rules::apply(state, 3);
  Rules::apply(state, 4);
  Rules::apply(state, 3);
  return state;
}

PolicyTensor make_policy(int move1, int move2) {
  PolicyTensor tensor;
  tensor.setZero();
  tensor(move1) = 1;
  tensor(move2) = 1;
  return tensor;
}

const std::string init_state_repr =
  "| | | | | | | |\n"
  "| | | | | | | |\n"
  "| | | | | | | |\n"
  "| | | | | | | |\n"
  "| | | |R| | | |\n"
  "| | | |R|Y| | |\n";

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
  for (int i = 0; i < 6; ++i) {
    repr += lines[i];
    repr += '\n';
  }

  return repr;
}

TEST(Symmetry, identity) {
  State state = make_init_state();

  group::element_t sym = groups::D1::kIdentity;
  group::element_t inv_sym = groups::D1::inverse(sym);
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr = init_state_repr;

  EXPECT_EQ(repr, expected_repr);
  Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym, 0);
  PolicyTensor expected_policy = make_policy(0, 1);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym, 0);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, flip) {
  State state = make_init_state();

  group::element_t sym = groups::D1::kFlip;
  group::element_t inv_sym = groups::D1::inverse(sym);
  Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | |R| | | |\n"
    "| | |Y|R| | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Symmetries::apply(policy, sym, 0);
  PolicyTensor expected_policy = make_policy(5, 6);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Symmetries::apply(policy, inv_sym, 0);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(PerfectOracle, query) {
  c4::PerfectOracle oracle;
  c4::PerfectOracle::MoveHistory move_history;
  c4::PerfectOracle::QueryResult result;

  result = oracle.query(move_history);
  EXPECT_EQ(result.best_score, 21);
  EXPECT_EQ(result.scores[0], -20);
  EXPECT_EQ(result.scores[1], -21);
  EXPECT_EQ(result.scores[2], 0);
  EXPECT_EQ(result.scores[3], 21);
  EXPECT_EQ(result.scores[4], 0);
  EXPECT_EQ(result.scores[5], -21);
  EXPECT_EQ(result.scores[6], -20);

  move_history.append(3);
  result = oracle.query(move_history);
  EXPECT_EQ(result.best_score, -20);
  EXPECT_EQ(result.scores[0], -17);
  EXPECT_EQ(result.scores[1], -19);
  EXPECT_EQ(result.scores[2], -19);
  EXPECT_EQ(result.scores[3], -20);
  EXPECT_EQ(result.scores[4], -19);
  EXPECT_EQ(result.scores[5], -19);
  EXPECT_EQ(result.scores[6], -17);

  move_history.append(3);
  move_history.append(3);
  move_history.append(3);
  move_history.append(3);
  move_history.append(3);
  result = oracle.query(move_history);
  EXPECT_EQ(result.best_score, 18);
  EXPECT_EQ(result.scores[0], -18);
  EXPECT_EQ(result.scores[1], 0);
  EXPECT_EQ(result.scores[2], 18);
  EXPECT_EQ(result.scores[3], c4::PerfectOracle::QueryResult::kIllegalMoveScore);
  EXPECT_EQ(result.scores[4], 18);
  EXPECT_EQ(result.scores[5], 0);
  EXPECT_EQ(result.scores[6], -18);
}

// PerfectOracle reuses a std::vector<std::string> buffer in-between query() calls. So it's worth
// repeating the test cases of TEST(PerfectOracle, query) in a different order to stress-test the
// correctness of the buffer reuse.
TEST(PerfectOracle, query2) {
  c4::PerfectOracle oracle;
  c4::PerfectOracle::MoveHistory move_history;
  c4::PerfectOracle::QueryResult result;

  move_history.append(3);
  result = oracle.query(move_history);
  EXPECT_EQ(result.best_score, -20);
  EXPECT_EQ(result.scores[0], -17);
  EXPECT_EQ(result.scores[1], -19);
  EXPECT_EQ(result.scores[2], -19);
  EXPECT_EQ(result.scores[3], -20);
  EXPECT_EQ(result.scores[4], -19);
  EXPECT_EQ(result.scores[5], -19);
  EXPECT_EQ(result.scores[6], -17);

  move_history.append(3);
  move_history.append(3);
  move_history.append(3);
  move_history.append(3);
  move_history.append(3);
  result = oracle.query(move_history);
  EXPECT_EQ(result.best_score, 18);
  EXPECT_EQ(result.scores[0], -18);
  EXPECT_EQ(result.scores[1], 0);
  EXPECT_EQ(result.scores[2], 18);
  EXPECT_EQ(result.scores[3], c4::PerfectOracle::QueryResult::kIllegalMoveScore);
  EXPECT_EQ(result.scores[4], 18);
  EXPECT_EQ(result.scores[5], 0);
  EXPECT_EQ(result.scores[6], -18);

  move_history.reset();
  result = oracle.query(move_history);
  EXPECT_EQ(result.best_score, 21);
  EXPECT_EQ(result.scores[0], -20);
  EXPECT_EQ(result.scores[1], -21);
  EXPECT_EQ(result.scores[2], 0);
  EXPECT_EQ(result.scores[3], 21);
  EXPECT_EQ(result.scores[4], 0);
  EXPECT_EQ(result.scores[5], -21);
  EXPECT_EQ(result.scores[6], -20);
}

TEST(Analyze, FromInitState) {
  State state;
  Rules::init_state(state);

  auto valid_masks = Rules::analyze(state).valid_moves();
  EXPECT_TRUE(valid_masks.count() == c4::kNumColumns);
}

TEST(Move, RoundTrip) {
  State state;
  Rules::init_state(state);

  // Column 0 serializes as "1", column 6 as "7"
  for (int col = 0; col < c4::kNumColumns; ++col) {
    c4::Move m(col);
    std::string s = m.to_str();
    EXPECT_EQ(s, std::to_string(col + 1)) << "col=" << col;
    c4::Move back = c4::Move::from_str(state, s);
    EXPECT_EQ(back, m) << "round-trip failed for col=" << col;
  }
}

TEST(Rules, ApplyAndCount) {
  State state;
  Rules::init_state(state);

  EXPECT_EQ(Rules::analyze(state).valid_moves().count(), c4::kNumColumns);

  Rules::apply(state, 0);
  EXPECT_EQ(Rules::analyze(state).valid_moves().count(), c4::kNumColumns);  // no column filled

  // Fill column 0 completely (6 rows)
  for (int i = 1; i < c4::kNumRows; ++i) {
    Rules::apply(state, 1);  // Y filler in col 1
    Rules::apply(state, 0);  // R fills col 0
  }
  // Col 0 is now full; valid moves should be kNumColumns - 1
  auto result = Rules::analyze(state);
  EXPECT_FALSE(result.is_terminal());
  EXPECT_EQ(result.valid_moves().count(), c4::kNumColumns - 1);
  EXPECT_FALSE(result.valid_moves().contains(c4::Move(0)));
}

TEST(Rules, WinHorizontal) {
  // Red fills bottom row of cols 0-3; Yellow fills col 4 three times
  // Sequence (R then Y alternating): R@0, Y@4, R@1, Y@4, R@2, Y@4, R@3
  State state;
  Rules::init_state(state);
  Rules::apply(state, 0);  // R
  Rules::apply(state, 4);  // Y
  Rules::apply(state, 1);  // R
  Rules::apply(state, 4);  // Y
  Rules::apply(state, 2);  // R
  Rules::apply(state, 4);  // Y
  Rules::apply(state, 3);  // R -> 4 in a row horizontally

  auto result = Rules::analyze(state);
  EXPECT_TRUE(result.is_terminal());
  EXPECT_EQ(result.outcome()(c4::kRed), 1.0f);
  EXPECT_EQ(result.outcome()(c4::kYellow), 0.0f);
}

TEST(Rules, WinVertical) {
  // Red fills col 0 four times; Yellow fills col 1 three times interleaved
  // Sequence: R@0, Y@1, R@0, Y@1, R@0, Y@1, R@0
  State state;
  Rules::init_state(state);
  Rules::apply(state, 0);  // R
  Rules::apply(state, 1);  // Y
  Rules::apply(state, 0);  // R
  Rules::apply(state, 1);  // Y
  Rules::apply(state, 0);  // R
  Rules::apply(state, 1);  // Y
  Rules::apply(state, 0);  // R -> 4 in a column

  auto result = Rules::analyze(state);
  EXPECT_TRUE(result.is_terminal());
  EXPECT_EQ(result.outcome()(c4::kRed), 1.0f);
  EXPECT_EQ(result.outcome()(c4::kYellow), 0.0f);
}

TEST(Rules, WinDiagonal) {
  // Build an ascending diagonal: Red at (col=0,row=0),(col=1,row=1),(col=2,row=2),(col=3,row=3)
  // Pre-fill col 1 with 1 Y, col 2 with 2 Y, col 3 with 3 Y (as fillers)
  // Sequence: R@0, Y@1, R@4(filler), Y@2, R@5(filler), Y@3, R@6(filler),
  //           Y@2, R@1, Y@3, R@4(filler), Y@3, R@2, Y@6(filler), R@3
  State state;
  Rules::init_state(state);
  Rules::apply(state, 0);  // R@col0 -> row 0
  Rules::apply(state, 1);  // Y@col1 filler
  Rules::apply(state, 4);  // R@col4 filler
  Rules::apply(state, 2);  // Y@col2 filler
  Rules::apply(state, 5);  // R@col5 filler
  Rules::apply(state, 3);  // Y@col3 filler
  Rules::apply(state, 6);  // R@col6 filler
  Rules::apply(state, 2);  // Y@col2 2nd filler
  Rules::apply(state, 1);  // R@col1 -> row 1 (1 Y already there)
  Rules::apply(state, 3);  // Y@col3 2nd filler
  Rules::apply(state, 4);  // R@col4 2nd filler
  Rules::apply(state, 3);  // Y@col3 3rd filler
  Rules::apply(state, 2);  // R@col2 -> row 2 (2 Y already there)
  Rules::apply(state, 6);  // Y@col6 filler
  Rules::apply(state, 3);  // R@col3 -> row 3 (3 Y already there) => diagonal win

  auto result = Rules::analyze(state);
  EXPECT_TRUE(result.is_terminal());
  EXPECT_EQ(result.outcome()(c4::kRed), 1.0f);
  EXPECT_EQ(result.outcome()(c4::kYellow), 0.0f);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
