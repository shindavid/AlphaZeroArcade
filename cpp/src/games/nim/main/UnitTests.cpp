#include "games/nim/Game.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using Game = nim::Game;
using State = Game::State;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;
using SymmetryGroup = groups::TrivialGroup;
using GameResults = core::WinShareResults<Game::Constants::kNumPlayers>;
using InputTensorizor = core::InputTensorizor<Game>;

TEST(NimGameTest, InitialState) {
  State state;
  Rules::init_state(state);

  EXPECT_EQ(Rules::get_current_player(state), 0);
  EXPECT_EQ(state.stones_left, 21);  // Assuming the game starts with 21 stones
}

TEST(NimGameTest, MakeMove) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, nim::kTake3);

  EXPECT_EQ(state.stones_left, 18);
  EXPECT_EQ(Rules::get_current_player(state), 1);
}

TEST(NimGameTest, Player0Wins) {
  State state;
  Rules::init_state(state);
  std::vector<core::action_t> actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                                         nim::kTake3, nim::kTake3, nim::kTake3};

  for (core::action_t action : actions) {
    Rules::apply(state, action);
  }

  core::action_t last_action = actions.back();

  GameResults::Tensor outcome;
  bool terminal = Rules::is_terminal(state, 1 - state.current_player, last_action, outcome);

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[0], 1);
}

TEST(NimGameTest, Player1Wins) {
  State state;
  Rules::init_state(state);
  std::vector<core::action_t> actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                                         nim::kTake3, nim::kTake3, nim::kTake1, nim::kTake2};

  for (core::action_t action : actions) {
    Rules::apply(state, action);
  }

  GameResults::Tensor outcome;
  core::action_t last_action = actions.back();
  bool terminal = Rules::is_terminal(state, 1 - state.current_player, last_action, outcome);

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[1], 1);
}

TEST(NimGameTest, InvalidMove) {
  State state;
  Rules::init_state(state);
  EXPECT_THROW(Rules::apply(state, -1), std::invalid_argument);
  EXPECT_THROW(Rules::apply(state, 3), std::invalid_argument);
}

TEST(NimGameTest, tensorize) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, 1);  // Player 0
  Rules::apply(state, 0);  // Player 1

  InputTensorizor input_tensorizor;
  input_tensorizor.update(state);

  InputTensorizor::Tensor tensor = input_tensorizor.tensorize();
  float expectedValues[] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  for (int i = 0; i < tensor.size(); i++) {
    EXPECT_EQ(tensor.data()[i], expectedValues[i]);
  }
}

int main(int argc, char **argv) { return launch_gtest(argc, argv); }
