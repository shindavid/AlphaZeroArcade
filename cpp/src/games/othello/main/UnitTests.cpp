#include <core/tests/Common.hpp>
#include <games/othello/Constants.hpp>
#include <games/othello/Game.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

/*
 * Tests othello symmetry classes.
 */

using Game = othello::Game;
using Common = core::tests::Common<Game>;
using State = Game::State;
using StateHistory = Game::StateHistory;
using Policy = Game::Types::Policy;
using IO = Game::IO;
using Rules = Game::Rules;

State make_init_state() {
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, othello::kD3);
  return history.current();
}

Policy make_policy(int move) {
  Policy policy;
  auto& subpolicy = std::get<0>(policy);
  subpolicy.setZero();
  subpolicy(move) = 1;
  return subpolicy;
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

  Policy init_policy = make_policy(othello::kA3);
  Policy policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  Policy expected_policy = make_policy(othello::kA3);
  EXPECT_TRUE(Common::policies_match(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(Common::policies_match(policy, init_policy));
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

  Policy init_policy = make_policy(othello::kA3);
  Policy policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  Policy expected_policy = make_policy(othello::kF1);
  EXPECT_TRUE(Common::policies_match(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(Common::policies_match(policy, init_policy));
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

  Policy init_policy = make_policy(othello::kA3);
  Policy policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  Policy expected_policy = make_policy(othello::kH6);
  EXPECT_TRUE(Common::policies_match(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(Common::policies_match(policy, init_policy));
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

  Policy init_policy = make_policy(othello::kA3);
  Policy policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  Policy expected_policy = make_policy(othello::kC8);
  EXPECT_TRUE(Common::policies_match(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(Common::policies_match(policy, init_policy));
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

  Policy init_policy = make_policy(othello::kA3);
  Policy policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  Policy expected_policy = make_policy(othello::kA6);
  EXPECT_TRUE(Common::policies_match(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(Common::policies_match(policy, init_policy));
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

  Policy init_policy = make_policy(othello::kA3);
  Policy policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  Policy expected_policy = make_policy(othello::kH3);
  EXPECT_TRUE(Common::policies_match(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(Common::policies_match(policy, init_policy));
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

  Policy init_policy = make_policy(othello::kA3);
  Policy policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  Policy expected_policy = make_policy(othello::kC1);
  EXPECT_TRUE(Common::policies_match(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(Common::policies_match(policy, init_policy));
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

  Policy init_policy = make_policy(othello::kA3);
  Policy policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  Policy expected_policy = make_policy(othello::kF8);
  EXPECT_TRUE(Common::policies_match(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(Common::policies_match(policy, init_policy));
}

TEST(Symmetry, action_transforms) {
  Common::gtest_action_transforms();
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
