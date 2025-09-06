#include "alphazero/ManagerParams.hpp"
#include "alphazero/Node.hpp"
#include "core/tests/Common.hpp"
#include "games/connect4/Constants.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/PerfectOracle.hpp"
#include "util/CppUtil.hpp"
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
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

State make_init_state() {
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, 3);
  Rules::apply(history, 4);
  Rules::apply(history, 3);
  return history.current();
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
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr = init_state_repr;

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym, 0);
  PolicyTensor expected_policy = make_policy(0, 1);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym, 0);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, flip) {
  State state = make_init_state();

  group::element_t sym = groups::D1::kFlip;
  group::element_t inv_sym = groups::D1::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | |R| | | |\n"
    "| | |Y|R| | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym, 0);
  PolicyTensor expected_policy = make_policy(5, 6);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym, 0);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, action_transforms) { core::tests::Common<Game>::gtest_action_transforms(); }

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

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
