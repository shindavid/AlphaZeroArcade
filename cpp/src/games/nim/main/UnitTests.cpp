#include <gtest/gtest.h>
#include <games/nim/Game.hpp>


using Game = nim::Game;
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;
using SymmetryGroup = groups::TrivialGroup;
using GameResults = core::WinLossDrawResults;
using Types = core::GameTypes<Game::Constants, State, GameResults, SymmetryGroup>;

TEST(NimGameTest, InitialState) {
  StateHistory history;
  history.initialize(Rules{});
  State state = history.current();

  EXPECT_EQ(Rules::get_current_player(state), 0);
  EXPECT_EQ(state.stones_left, 21); // Assuming the game starts with 21 stones
}

TEST(NimGameTest, MakeMove) {
  StateHistory history;
  history.initialize(Rules{});
  Rules::apply(history, 3);
  State state = history.current();

  EXPECT_EQ(state.stones_left, 18);
  EXPECT_EQ(Rules::get_current_player(state), 1);
}

TEST(NimGameTest, Player0Wins) {
  StateHistory history;
  history.initialize(Rules{});
  Rules::apply(history, 3); // Player 0
  Rules::apply(history, 3); // Player 1
  Rules::apply(history, 3); // Player 0
  Rules::apply(history, 3); // Player 1
  Rules::apply(history, 3); // Player 0
  Rules::apply(history, 3); // Player 1
  Types::ActionOutcome outcome = Rules::apply(history, 3); // Player 0
  EXPECT_EQ(outcome.terminal, true);
  EXPECT_EQ(outcome.terminal_tensor[0], 1);
}

TEST(NimGameTest, Player1Wins) {
  StateHistory history;
  history.initialize(Rules{});
  Rules::apply(history, 3); // Player 0
  Rules::apply(history, 3); // Player 1
  Rules::apply(history, 3); // Player 0
  Rules::apply(history, 3); // Player 1
  Rules::apply(history, 3); // Player 0
  Rules::apply(history, 3); // Player 1
  Rules::apply(history, 1); // Player 0
  Types::ActionOutcome outcome = Rules::apply(history, 2); // Player 1
  EXPECT_EQ(outcome.terminal, true);
  EXPECT_EQ(outcome.terminal_tensor[1], 1);
}

TEST(NimGameTest, InvalidMove) {
  StateHistory history;
  history.initialize(Rules{});
  EXPECT_THROW(Rules::apply(history, 0), std::invalid_argument);
  EXPECT_THROW(Rules::apply(history, 4), std::invalid_argument);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}