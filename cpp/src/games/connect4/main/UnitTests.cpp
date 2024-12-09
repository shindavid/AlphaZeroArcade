#include <core/tests/Common.hpp>
#include <games/connect4/Constants.hpp>
#include <games/connect4/Game.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchThread.hpp>
#include <mcts/SharedData.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>
#include <util/MetaProgramming.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

/*
 * Tests connect4 symmetry classes.
 */

using Game = c4::Game;
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensorVariant = Game::Types::PolicyTensorVariant;
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

PolicyTensorVariant make_policy(int move1, int move2) {
  PolicyTensorVariant policy;
  mp::TypeAt_t<PolicyTensorVariant, 0> tensor = std::get<0>(policy);
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

bool policies_match(const PolicyTensorVariant& p1, const PolicyTensorVariant& p2) {
  return eigen_util::equal(std::get<0>(p1), std::get<0>(p2));
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

  PolicyTensorVariant init_policy = make_policy(0, 1);
  PolicyTensorVariant policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensorVariant expected_policy = make_policy(0, 1);
  EXPECT_TRUE(policies_match(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(policies_match(policy, init_policy));
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

  PolicyTensorVariant init_policy = make_policy(0, 1);
  PolicyTensorVariant policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensorVariant expected_policy = make_policy(5, 6);
  EXPECT_TRUE(policies_match(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(policies_match(policy, init_policy));
}

TEST(Symmetry, action_transforms) {
  core::tests::Common<Game>::gtest_action_transforms();
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
