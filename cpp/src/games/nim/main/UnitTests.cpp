#include <games/nim/Game.hpp>

#include <gtest/gtest.h>

#include <iostream>

using Game = nim::Game;
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;
using Types = Game::Types;
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

TEST(NimGameTest, VerifyChanceStatus) {
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, nim::kTake3);
  State state = history.current();
  if (nim::kChanceDistributionSize == 0) {
    EXPECT_FALSE(state.chance_active);
    EXPECT_EQ(Rules::get_action_mode(state), 0);
    EXPECT_FALSE(Rules::is_chance_mode(Rules::get_action_mode(state)));
  } else {
    EXPECT_TRUE(state.chance_active);
    EXPECT_EQ(Rules::get_action_mode(state), 1);
    EXPECT_TRUE(Rules::is_chance_mode(Rules::get_action_mode(state)));
  }
}

TEST(NimGameTest, VerifyDistFailure) {
  StateHistory history;
  history.initialize(Rules{});

  EXPECT_THROW(Rules::get_chance_distribution(history.current()), std::invalid_argument);
}

TEST(NimGameTest, VerifyDist) {
  if (nim::kChanceDistributionSize == 0) {
    return;
  }
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, nim::kTake3);
  State state = history.current();

  PolicyTensor dist = Rules::get_chance_distribution(state);

  for (int i = 0; i < nim::kChanceDistributionSize; ++i) {
    EXPECT_EQ(dist[i], 1.0 / nim::kChanceDistributionSize);
  }

  for (int i = nim::kChanceDistributionSize; i < dist.size(); ++i) {
    EXPECT_EQ(dist[i], 0);
  }
}

TEST(NimGameTest, ChanceMove) {
  if (nim::kChanceDistributionSize == 0) {
    return;
  }
  int num_trials = 1000;
  float sum = 0;
  for (int i = 0; i < num_trials; i++) {
    StateHistory history;
    history.initialize(Rules{});

    Rules::apply(history, nim::kTake3);

    Types::PolicyTensor dist = Rules::get_chance_distribution(history.current());
    core::action_t chance_action = eigen_util::sample(dist);
    Rules::apply(history, chance_action);

    sum += history.current().stones_left;
  }
  EXPECT_NEAR(sum / num_trials, 17.5, 3 * std::sqrt(0.5 * 0.5 / num_trials));
}

TEST(NimGameTest, Player0Wins) {
  StateHistory history;
  history.initialize(Rules{});
  std::vector<core::action_t> actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                                         nim::kTake3, nim::kTake3, nim::kTake3};

  for (core::action_t action : actions) {
    Rules::apply(history, action);
    // chance action
    Rules::apply(history, 0);
  }

  core::action_t last_action = actions.back();

  GameResults::Tensor outcome;
  bool terminal = Rules::is_terminal(history.current(), 1 - history.current().next_player,
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
    // chance action
    Rules::apply(history, 0);
  }

  GameResults::Tensor outcome;
  core::action_t last_action = actions.back();
  bool terminal = Rules::is_terminal(history.current(), 1 - history.current().stones_left,
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
  Rules::apply(history, 1);  // Player 0 removes 2 stones
  Rules::apply(history, 0);  // chance
  Rules::apply(history, 0);  // Player 1 removes 1 stone
  Rules::apply(history, 0);  // chance

  Game::InputTensorizor::Tensor tensor =
      Game::InputTensorizor::tensorize(history.begin(), history.end() - 1);
  float expectedValues[] = {0, 1, 0, 0, 1, 0, 0};
  for (int i = 0; i < tensor.size(); i++) {
    EXPECT_EQ(tensor.data()[i], expectedValues[i]);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
