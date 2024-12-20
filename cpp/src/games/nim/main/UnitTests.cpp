#include <games/nim/Game.hpp>

#include <gtest/gtest.h>

#include <iostream>

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
  EXPECT_EQ(state.get_stones(), 21);  // Assuming the game starts with 21 stones
}

TEST(NimGameTest, MakeMove) {
  StateHistory history;
  history.initialize(Rules{});
  Rules::apply(history, nim::kTake3);
  State state = history.current();

  EXPECT_EQ(state.get_stones(), 18);
  EXPECT_EQ(Rules::get_current_player(state), 1);
}

TEST(NimGameTest, VerifyChanceStatus) {
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, nim::kTake3);
  State state = history.current();
  if (nim::kMaxRandomStonesToTake == 0) {
    EXPECT_TRUE(state.is_player_ready());
    EXPECT_EQ(Rules::get_action_mode(state), 0);
    EXPECT_FALSE(Rules::prior_prob_known(state));
  } else {
    EXPECT_FALSE(state.is_player_ready());
    EXPECT_EQ(Rules::get_action_mode(state), 1);
    EXPECT_TRUE(Rules::prior_prob_known(state));
  }
}

TEST(NimGameTest, VerifyDistFailure) {
  StateHistory history;
  history.initialize(Rules{});

  EXPECT_THROW(Rules::get_prior_prob(history.current()), std::invalid_argument);
}

TEST(NimGameTest, VerifyDist) {
  if (nim::kMaxRandomStonesToTake == 0) {
    return;
  }
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, nim::kTake3);
  State state = history.current();

  PolicyTensor dist = Rules::get_prior_prob(state);

  for (int i = 0; i < nim::kMaxRandomStonesToTake + 1; ++i) {
    EXPECT_EQ(dist[i], 1.0 / (nim::kMaxRandomStonesToTake + 1));
  }

  for (int i = nim::kMaxRandomStonesToTake + 1; i < dist.size(); ++i) {
    EXPECT_EQ(dist[i], 0);
  }
}

TEST(NimGameTest, ChanceMove) {
  if (nim::kMaxRandomStonesToTake == 0) {
    return;
  }
  int num_trials = 1000;
  float sum = 0;
  for (int i = 0; i < num_trials; i++) {
    StateHistory history;
    history.initialize(Rules{});

    Rules::apply(history, nim::kTake3);

    core::action_t chance_action = Rules::sample_chance_action(history);
    Rules::apply(history, chance_action);

    sum += history.current().get_stones();
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
  bool terminal = Rules::is_terminal(history.current(), 1 - history.current().get_player(),
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
  bool terminal = Rules::is_terminal(history.current(), 1 - history.current().get_player(),
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
  Rules::apply(history, 0);  // chance
  Rules::apply(history, 0);  // Player 1
  Rules::apply(history, 0);  // chance

  Game::InputTensorizor::Tensor tensor =
      Game::InputTensorizor::tensorize(history.begin(), history.end() - 1);
  float expectedValues[] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  for (int i = 0; i < tensor.size(); i++) {
    EXPECT_EQ(tensor.data()[i], expectedValues[i]);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
