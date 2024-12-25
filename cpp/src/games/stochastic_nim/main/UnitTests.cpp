#include <games/stochastic_nim/Game.hpp>

#include <gtest/gtest.h>

#include <iostream>

using Game = stochastic_nim::Game;
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;
using Types = Game::Types;
using SymmetryGroup = groups::TrivialGroup;
using GameResults = core::WinShareResults<Game::Constants::kNumPlayers>;

TEST(StochasticNimGameTest, InitialState) {
  StateHistory history;
  history.initialize(Rules{});
  State state = history.current();

  EXPECT_EQ(Rules::get_current_player(state), 0);
  EXPECT_EQ(state.stones_left, 21);  // Assuming the game starts with 21 stones
}

TEST(StochasticNimGameTest, MakeMove) {
  StateHistory history;
  history.initialize(Rules{});
  Rules::apply(history, stochastic_nim::kTake3);
  State state = history.current();

  EXPECT_EQ(state.stones_left, 18);
  EXPECT_EQ(Rules::get_current_player(state), 0);
}

TEST(StochasticNimGameTest, VerifyChanceStatus) {
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, stochastic_nim::kTake3);
  State state = history.current();
  if (stochastic_nim::kChanceDistributionSize == 0) {
    EXPECT_EQ(state.current_mode, stochastic_nim::kPlayerMode);
    EXPECT_EQ(Rules::get_action_mode(state), 0);
    EXPECT_FALSE(Rules::is_chance_mode(Rules::get_action_mode(state)));
  } else {
    EXPECT_EQ(state.current_mode, stochastic_nim::kChanceMode);
    EXPECT_EQ(Rules::get_action_mode(state), 1);
    EXPECT_TRUE(Rules::is_chance_mode(Rules::get_action_mode(state)));
  }
}

TEST(StochasticNimGameTest, VerifyDistFailure) {
  StateHistory history;
  history.initialize(Rules{});

  EXPECT_THROW(Rules::get_chance_distribution(history.current()), std::invalid_argument);
}

TEST(StochasticNimGameTest, VerifyDist) {
  if (stochastic_nim::kChanceDistributionSize == 0) {
    return;
  }
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, stochastic_nim::kTake3);
  State state = history.current();

  PolicyTensor dist = Rules::get_chance_distribution(state);

  EXPECT_NEAR(dist(0), 0.2, 1e-6);
  EXPECT_NEAR(dist(1), 0.3, 1e-6);
  EXPECT_NEAR(dist(2), 0.5, 1e-6);
}

TEST(StochasticNimGameTest, ChanceMove) {
  if (stochastic_nim::kChanceDistributionSize == 0) {
    return;
  }
  int num_trials = 1000;
  float sum = 0;
  for (int i = 0; i < num_trials; i++) {
    StateHistory history;
    history.initialize(Rules{});

    Rules::apply(history, stochastic_nim::kTake3);

    Types::PolicyTensor dist = Rules::get_chance_distribution(history.current());
    core::action_t chance_action = eigen_util::sample(dist);
    Rules::apply(history, chance_action);

    sum += history.current().stones_left;
  }
  float mean = 18 * 0.2 + 17 * 0.3 + 16 * 0.5;
  float sigma = std::sqrt(
      (0.2 * std::pow(16 - mean, 2) + 0.3 * std::pow(17 - mean, 2) + 0.5 * std::pow(18 - mean, 2)) /
      num_trials);
  EXPECT_NEAR(sum / num_trials, mean, 3 * sigma);
}

TEST(StochasticNimGameTest, Player0Wins) {
  StateHistory history;
  history.initialize(Rules{});
  std::vector<core::action_t> actions = {stochastic_nim::kTake3, stochastic_nim::kTake3,
                                         stochastic_nim::kTake3, stochastic_nim::kTake3,
                                         stochastic_nim::kTake3, stochastic_nim::kTake3,
                                         stochastic_nim::kTake3};

  for (core::action_t action : actions) {
    Rules::apply(history, action);
    // chance action
    Rules::apply(history, 0);
  }

  core::action_t last_action = actions.back();

  GameResults::Tensor outcome;
  bool terminal = Rules::is_terminal(history.current(), 1 - history.current().current_player,
                                     last_action, outcome);

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[0], 1);
}

TEST(StochasticNimGameTest, Player1Wins) {
  StateHistory history;
  history.initialize(Rules{});
  std::vector<core::action_t> actions = {stochastic_nim::kTake3, stochastic_nim::kTake3,
                                         stochastic_nim::kTake3, stochastic_nim::kTake3,
                                         stochastic_nim::kTake3, stochastic_nim::kTake3,
                                         stochastic_nim::kTake1, stochastic_nim::kTake2};

  for (core::action_t action : actions) {
    Rules::apply(history, action);
    // chance action
    Rules::apply(history, 0);
  }

  GameResults::Tensor outcome;
  core::action_t last_action = actions.back();
  bool terminal = Rules::is_terminal(history.current(), 1 - history.current().current_player,
                                     last_action, outcome);

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[1], 1);
}

TEST(StochasticNimGameTest, InvalidMove) {
  StateHistory history;
  history.initialize(Rules{});
  EXPECT_THROW(Rules::apply(history, -1), std::invalid_argument);
  EXPECT_THROW(Rules::apply(history, 3), std::invalid_argument);
}

TEST(StochasticNimGameTest, MoveProbMass) {
  StateHistory history;
  State state;
  state.stones_left = 1;
  state.current_player = 0;
  state.current_mode = stochastic_nim::kChanceMode;
  history.update(state);
  Game::Types::ChanceDistribution dist = Rules::get_chance_distribution(state);

  EXPECT_NEAR(dist(0), 0.2, 1e-6);
  EXPECT_NEAR(dist(1), 0.8, 1e-6);
  EXPECT_NEAR(dist(2), 0.0, 1e-6);
}

TEST(StochasticNimGameTest, tensorize) {
  StateHistory history;
  history.initialize(Rules{});
  Rules::apply(history, stochastic_nim::kTake2);  // Player 0 removes 2 stones
  Rules::apply(history, 0);  // chance
  Rules::apply(history, stochastic_nim::kTake1);  // Player 1 removes 1 stone
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

