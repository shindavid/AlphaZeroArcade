#include <gtest/gtest.h>
#include <games/nim/Game.hpp>


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
  EXPECT_EQ(state.stones_left, 21); // Assuming the game starts with 21 stones
}

TEST(NimGameTest, MakeMove) {
  StateHistory history;
  history.initialize(Rules{});
  Rules::apply(history, 2);
  State state = history.current();

  EXPECT_EQ(state.stones_left, 18);
  EXPECT_EQ(Rules::get_current_player(state), 1);
}

TEST(NimGameTest, Player0Wins) {
  StateHistory history;
  history.initialize(Rules{});
  Rules::apply(history, 2); // Player 0
  Rules::apply(history, 2); // Player 1
  Rules::apply(history, 2); // Player 0
  Rules::apply(history, 2); // Player 1
  Rules::apply(history, 2); // Player 0
  Rules::apply(history, 2); // Player 1
  Rules::apply(history, 2); // Player 0

  GameResults::Tensor outcome;
  core::action_t last_action = 2;
  bool isTerminal = Rules::is_terminal(history.current(), 1 - history.current().current_player,
                                       last_action, outcome);

  EXPECT_EQ(isTerminal, true);
  EXPECT_EQ(outcome[0], 1);
}

TEST(NimGameTest, Player1Wins) {
  StateHistory history;
  history.initialize(Rules{});
  Rules::apply(history, 2); // Player 0
  Rules::apply(history, 2); // Player 1
  Rules::apply(history, 2); // Player 0
  Rules::apply(history, 2); // Player 1
  Rules::apply(history, 2); // Player 0
  Rules::apply(history, 2); // Player 1
  Rules::apply(history, 0); // Player 0
  Rules::apply(history, 1); // Player 1

  GameResults::Tensor outcome;
  core::action_t last_action = 2;
  bool isTerminal = Rules::is_terminal(history.current(), 1 - history.current().current_player,
                                       last_action, outcome);

  EXPECT_EQ(isTerminal, true);
  EXPECT_EQ(outcome[1], 1);
}

TEST(NimGameTest, InvalidMove) {
  StateHistory history;
  history.initialize(Rules{});
  EXPECT_THROW(Rules::apply(history, -1), std::invalid_argument);
  EXPECT_THROW(Rules::apply(history, 3), std::invalid_argument);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}