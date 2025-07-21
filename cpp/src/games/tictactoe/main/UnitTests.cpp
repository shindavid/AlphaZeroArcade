#include "core/tests/Common.hpp"
#include "games/tictactoe/Constants.hpp"
#include "games/tictactoe/Game.hpp"
#include "util/CppUtil.hpp"
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
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

template<typename... Ts>
State make_state(Ts... moves) {
  StateHistory history;
  history.initialize(Rules{});

  for (int move : {moves...}) {
    Rules::apply(history, move);
  }
  return history.current();
}

State make_init_state() {
  return make_state(7, 2);
}

PolicyTensor make_policy(int move1, int move2) {
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

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(0, 1);
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
      "0 1 2  | | | |\n"
      "3 4 5  |X| | |\n"
      "6 7 8  | | |O|\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(2, 5);
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
      "0 1 2  | |X| |\n"
      "3 4 5  | | | |\n"
      "6 7 8  |O| | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(7, 8);
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
      "0 1 2  |O| | |\n"
      "3 4 5  | | |X|\n"
      "6 7 8  | | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(3, 6);
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
      "0 1 2  | |X| |\n"
      "3 4 5  | | | |\n"
      "6 7 8  | | |O|\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(6, 7);
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
      "0 1 2  |O| | |\n"
      "3 4 5  | | | |\n"
      "6 7 8  | |X| |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(1, 2);
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
      "0 1 2  | | | |\n"
      "3 4 5  | | |X|\n"
      "6 7 8  |O| | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(0, 3);
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
      "0 1 2  | | |O|\n"
      "3 4 5  |X| | |\n"
      "6 7 8  | | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(5, 8);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, action_transforms) {
  core::tests::Common<Game>::gtest_action_transforms();
}

TEST(Symmetry, canonicalization) {
  State state = make_state(2, 1);

  std::string expected_repr =
      "0 1 2  | |O|X|\n"
      "3 4 5  | | | |\n"
      "6 7 8  | | | |\n";

  std::string repr = get_repr(state);

  EXPECT_EQ(repr, expected_repr);

  group::element_t e = Game::Symmetries::get_canonical_symmetry(state);
  EXPECT_EQ(e, groups::D4::kMirrorHorizontal);

  Game::Symmetries::apply(state, e);

  expected_repr =
      "0 1 2  |X|O| |\n"
      "3 4 5  | | | |\n"
      "6 7 8  | | | |\n";

  repr = get_repr(state);

  EXPECT_EQ(repr, expected_repr);
}

int main(int argc, char** argv) {
  return launch_gtest(argc, argv);
}
