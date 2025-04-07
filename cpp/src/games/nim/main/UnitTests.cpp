#include <games/nim/Game.hpp>
#include <util/GTestUtil.hpp>

#include <gtest/gtest.h>

using Game = nim::Game;
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;
using SymmetryGroup = groups::TrivialGroup;
using GameResults = core::WinShareResults<Game::Constants::kNumPlayers>;

TEST(NimGameTest, InitialState) {
  StateHistory history;
  history.initialize(Rules{});
  State state = history.current();

  EXPECT_EQ(Rules::get_current_player(state), 0);
  EXPECT_EQ(state.stones_left, 21);  // Assuming the game starts with 21 stones
}

TEST(NimGameTest, MakeMove) {
  StateHistory history;
  history.initialize(Rules{});
  Rules::apply(history, nim::kTake3);
  State state = history.current();

  EXPECT_EQ(state.stones_left, 18);
  EXPECT_EQ(Rules::get_current_player(state), 1);
}

TEST(NimGameTest, Player0Wins) {
  StateHistory history;
  history.initialize(Rules{});
  std::vector<core::action_t> actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                                         nim::kTake3, nim::kTake3, nim::kTake3};

  for (core::action_t action : actions) {
    Rules::apply(history, action);
  }

  core::action_t last_action = actions.back();

  GameResults::Tensor outcome;
  bool terminal = Rules::is_terminal(history.current(), 1 - history.current().current_player,
                                     last_action, outcome);

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[0], 1);
}

TEST(NimGameTest, Player1Wins) {
  StateHistory history;
  history.initialize(Rules{});
  std::vector<core::action_t> actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                                         nim::kTake3, nim::kTake3, nim::kTake1, nim::kTake2};

  for (core::action_t action : actions) {
    Rules::apply(history, action);
  }

  GameResults::Tensor outcome;
  core::action_t last_action = actions.back();
  bool terminal = Rules::is_terminal(history.current(), 1 - history.current().current_player,
                                     last_action, outcome);

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[1], 1);
}

TEST(NimGameTest, InvalidMove) {
  StateHistory history;
  history.initialize(Rules{});
  EXPECT_THROW(Rules::apply(history, -1), std::invalid_argument);
  EXPECT_THROW(Rules::apply(history, 3), std::invalid_argument);
}

TEST(NimGameTest, tensorize) {
  StateHistory history;
  history.initialize(Rules{});
  Rules::apply(history, 1);  // Player 0
  Rules::apply(history, 0);  // Player 1

  Game::InputTensorizor::Tensor tensor =
      Game::InputTensorizor::tensorize(history.begin(), history.end() - 1);
  float expectedValues[] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  for (int i = 0; i < tensor.size(); i++) {
    EXPECT_EQ(tensor.data()[i], expectedValues[i]);
  }
}

int main(int argc, char **argv) {
  return launch_gtest(argc, argv);
}
