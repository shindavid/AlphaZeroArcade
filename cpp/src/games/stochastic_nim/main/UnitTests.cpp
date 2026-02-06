#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/players/PerfectPlayer.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#include <iostream>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using Game = stochastic_nim::Game;
using State = Game::State;
using PolicyTensor = Game::Types::PolicyTensor;
using ActionRequest = core::ActionRequest<Game>;
using ChanceDistribution = Game::Types::ChanceDistribution;
using IO = Game::IO;
using Rules = Game::Rules;
using SymmetryGroup = groups::TrivialGroup;
using GameResults = core::WinShareResults<Game::Constants::kNumPlayers>;
using InputTensorizor = core::InputTensorizor<Game>;

class PerfectPlayerTest : public testing::Test {
 protected:
  using PerfectPlayer = stochastic_nim::PerfectPlayer;
  using PerfectStrategy = stochastic_nim::PerfectStrategy;
  using State = PerfectPlayer::base_t::State;
  using ActionMask = PerfectPlayer::ActionMask;

 public:
  PerfectPlayerTest() : player_(PerfectPlayer::Params(1, false), &strategy_) {}

  core::action_t get_action_response(const State& state) {
    ActionMask valid_actions;
    ActionRequest request(state, valid_actions);
    core::ActionResponse response = player_.get_action_response(request);
    return response.get_action();
  }

 private:
  PerfectStrategy strategy_;
  PerfectPlayer player_;
};

class PerfectStrategyTest : public testing::Test {
 protected:
  using PerfectStrategy = stochastic_nim::PerfectStrategy;

 public:
  PerfectStrategy get_strategy() { return PerfectStrategy(); }
};

TEST_F(PerfectPlayerTest, 4_stones_player0) {
  State state{4, stochastic_nim::kPlayer0, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 5_stones_player0) {
  State state{5, stochastic_nim::kPlayer0, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 6_stones_player0) {
  State state{6, stochastic_nim::kPlayer0, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake1);
}

TEST_F(PerfectPlayerTest, 4_stones_player1) {
  State state{4, stochastic_nim::kPlayer1, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 5_stones_player1) {
  State state{5, stochastic_nim::kPlayer1, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 6_stones_player1) {
  State state{6, stochastic_nim::kPlayer1, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake1);
}

TEST_F(PerfectPlayerTest, chance_mode_throw_error) {
  State state{4, stochastic_nim::kPlayer0, stochastic_nim::kChanceMode};
  EXPECT_THROW(get_action_response(state), util::Exception);
}

TEST_F(PerfectPlayerTest, greater_than_starting_stones_throw_error) {
  State state{10000, stochastic_nim::kPlayer0, stochastic_nim::kPlayerMode};
  EXPECT_THROW(get_action_response(state), util::Exception);
}

TEST_F(PerfectStrategyTest, verify_state_values) {
  PerfectStrategy strategy = get_strategy();
  EXPECT_NEAR(strategy.get_state_value_after(5), 0.16, 1e-6);
  EXPECT_NEAR(strategy.get_state_value_after(4), 0.04, 1e-6);
  EXPECT_NEAR(strategy.get_state_value_after(3), 0.0, 1e-6);
  EXPECT_NEAR(strategy.get_state_value_after(2), 0.5, 1e-6);
  EXPECT_NEAR(strategy.get_state_value_after(1), 0.8, 1e-6);
  EXPECT_NEAR(strategy.get_state_value_after(0), 1.0, 1e-6);
  EXPECT_EQ(strategy.get_optimal_action(6), stochastic_nim::kTake1);
  EXPECT_EQ(strategy.get_optimal_action(5), stochastic_nim::kTake3);
  EXPECT_EQ(strategy.get_optimal_action(4), stochastic_nim::kTake3);
  EXPECT_EQ(strategy.get_optimal_action(3), stochastic_nim::kTake3);
  EXPECT_EQ(strategy.get_optimal_action(2), stochastic_nim::kTake2);
  EXPECT_EQ(strategy.get_optimal_action(1), stochastic_nim::kTake1);
  EXPECT_THROW(strategy.get_optimal_action(0), util::Exception);
}

TEST(StochasticNimGameTest, InitialState) {
  State state;
  Rules::init_state(state);

  EXPECT_EQ(Rules::get_current_player(state), 0);
  EXPECT_EQ(state.stones_left, 21);  // Assuming the game starts with 21 stones
}

TEST(StochasticNimGameTest, MakeMove) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, stochastic_nim::kTake3);

  EXPECT_EQ(state.stones_left, 18);
  EXPECT_EQ(Rules::get_current_player(state), 0);
}

TEST(StochasticNimGameTest, VerifyChanceStatus) {
  State state;
  Rules::init_state(state);

  Rules::apply(state, stochastic_nim::kTake3);
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
  State state;
  Rules::init_state(state);

  EXPECT_THROW(Rules::get_chance_distribution(state), std::invalid_argument);
}

TEST(StochasticNimGameTest, VerifyDist) {
  if (stochastic_nim::kChanceDistributionSize == 0) {
    return;
  }
  State state;
  Rules::init_state(state);

  Rules::apply(state, stochastic_nim::kTake3);

  PolicyTensor dist = Rules::get_chance_distribution(state);

  EXPECT_NEAR(dist(0), 0.2, 1e-6);
  EXPECT_NEAR(dist(1), 0.3, 1e-6);
  EXPECT_NEAR(dist(2), 0.5, 1e-6);
}

TEST(StochasticNimGameTest, ChanceMove) {
  util::Random::set_seed(1);
  if (stochastic_nim::kChanceDistributionSize == 0) {
    return;
  }
  int num_trials = 1000;
  float sum = 0;
  for (int i = 0; i < num_trials; i++) {
    State state;
    Rules::init_state(state);

    Rules::apply(state, stochastic_nim::kTake3);

    PolicyTensor dist = Rules::get_chance_distribution(state);
    core::action_t chance_action = eigen_util::sample(dist);
    Rules::apply(state, chance_action);

    sum += state.stones_left;
  }
  float mean = 18 * 0.2 + 17 * 0.3 + 16 * 0.5;
  float sigma = std::sqrt(
    (0.2 * std::pow(16 - mean, 2) + 0.3 * std::pow(17 - mean, 2) + 0.5 * std::pow(18 - mean, 2)) /
    num_trials);
  EXPECT_NEAR(sum / num_trials, mean, 3 * sigma);
}

TEST(StochasticNimGameTest, Player0Wins) {
  State state;
  Rules::init_state(state);
  std::vector<core::action_t> actions = {
    stochastic_nim::kTake3, stochastic_nim::kTake3, stochastic_nim::kTake3, stochastic_nim::kTake3,
    stochastic_nim::kTake3, stochastic_nim::kTake3, stochastic_nim::kTake3};

  for (core::action_t action : actions) {
    Rules::apply(state, action);
    // chance action
    Rules::apply(state, 0);
  }

  core::action_t last_action = actions.back();

  GameResults::Tensor outcome;
  bool terminal = Rules::is_terminal(state, 1 - state.current_player, last_action, outcome);

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[0], 1);
}

TEST(StochasticNimGameTest, Player1Wins) {
  State state;
  Rules::init_state(state);
  std::vector<core::action_t> actions = {
    stochastic_nim::kTake3, stochastic_nim::kTake3, stochastic_nim::kTake3, stochastic_nim::kTake3,
    stochastic_nim::kTake3, stochastic_nim::kTake3, stochastic_nim::kTake1, stochastic_nim::kTake2};

  for (core::action_t action : actions) {
    Rules::apply(state, action);
    // chance action
    Rules::apply(state, 0);
  }

  GameResults::Tensor outcome;
  core::action_t last_action = actions.back();
  bool terminal = Rules::is_terminal(state, 1 - state.current_player, last_action, outcome);

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[1], 1);
}

TEST(StochasticNimGameTest, InvalidMove) {
  State state;
  Rules::init_state(state);
  EXPECT_THROW(Rules::apply(state, -1), std::invalid_argument);
  EXPECT_THROW(Rules::apply(state, 3), std::invalid_argument);
}

TEST(StochasticNimGameTest, MoveProbMass) {
  State state;
  state.stones_left = 1;
  state.current_player = 0;
  state.current_mode = stochastic_nim::kChanceMode;
  ChanceDistribution dist = Rules::get_chance_distribution(state);

  EXPECT_NEAR(dist(0), 0.2, 1e-6);
  EXPECT_NEAR(dist(1), 0.8, 1e-6);
  EXPECT_NEAR(dist(2), 0.0, 1e-6);
}

TEST(StochasticNimGameTest, tensorize) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, stochastic_nim::kTake2);  // Player 0 removes 2 stones
  Rules::apply(state, 0);                       // chance
  Rules::apply(state, stochastic_nim::kTake1);  // Player 1 removes 1 stone
  Rules::apply(state, 0);                       // chance

  InputTensorizor input_tensorizor;
  input_tensorizor.update(state);

  InputTensorizor::Tensor tensor = input_tensorizor.tensorize();
  float expectedValues[] = {0, 1, 0, 0, 1, 0, 0};
  for (int i = 0; i < tensor.size(); i++) {
    EXPECT_EQ(tensor.data()[i], expectedValues[i]);
  }
}

void print_perfect_strategy_info() {
  stochastic_nim::PerfectStrategy strategy;
  for (int i = stochastic_nim::kStartingStones; i > 0; --i) {
    std::cout << "Stones left: " << i << " Action: " << strategy.get_optimal_action(i) + 1
              << " V: " << strategy.get_state_value_before(i) << std::endl;

    for (int j = 0; j < stochastic_nim::kMaxStonesToTake; ++j) {
      std::cout << "  Take " << j + 1;
      if (i - j - 1 >= 0) {
        std::cout << " Value: " << strategy.get_state_value_after(i - j - 1) << std::endl;
      } else {
        std::cout << " Value: N/A" << std::endl;
      }
    }
  }
}

int main(int argc, char** argv) {
  print_perfect_strategy_info();
  return launch_gtest(argc, argv);
}
