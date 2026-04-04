#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "games/stochastic_nim/Constants.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/InputTensorizor.hpp"
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
using ActionResponse = core::ActionResponse<Game>;
using Move = Game::Move;
using MoveList = Game::MoveList;
using ChanceDistribution = Game::ChanceDistribution;
using IO = Game::IO;
using Rules = Game::Rules;
using SymmetryGroup = groups::TrivialGroup;
using GameResults = core::WinShareResults<Game::Constants::kNumPlayers>;
using InputTensorizor = stochastic_nim::InputTensorizor;

class PerfectPlayerTest : public testing::Test {
 protected:
  using PerfectPlayer = stochastic_nim::PerfectPlayer;
  using PerfectStrategy = stochastic_nim::PerfectStrategy;
  using State = PerfectPlayer::base_t::State;

 public:
  PerfectPlayerTest() : player_(PerfectPlayer::Params(1, false), &strategy_) {}

  int get_action_response(const State& state) {
    MoveList valid_moves;
    ActionRequest request(state, valid_moves);
    ActionResponse response = player_.get_action_response(request);
    return response.get_move().index();
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

TEST(Analyze, FromInitState) {
  State state;
  Rules::init_state(state);

  auto valid_masks = Rules::analyze(state).valid_moves();
  EXPECT_TRUE(valid_masks.size() == 3);
}

TEST_F(PerfectPlayerTest, 4_stones_player0) {
  State state{4, stochastic_nim::kPlayer0, stochastic_nim::kPlayer1, stochastic_nim::kPlayerPhase};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 5_stones_player0) {
  State state{5, stochastic_nim::kPlayer0, stochastic_nim::kPlayer1, stochastic_nim::kPlayerPhase};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 6_stones_player0) {
  State state{6, stochastic_nim::kPlayer0, stochastic_nim::kPlayer1, stochastic_nim::kPlayerPhase};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake1);
}

TEST_F(PerfectPlayerTest, 4_stones_player1) {
  State state{4, stochastic_nim::kPlayer1, stochastic_nim::kPlayer0, stochastic_nim::kPlayerPhase};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 5_stones_player1) {
  State state{5, stochastic_nim::kPlayer1, stochastic_nim::kPlayer0, stochastic_nim::kPlayerPhase};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 6_stones_player1) {
  State state{6, stochastic_nim::kPlayer1, stochastic_nim::kPlayer0, stochastic_nim::kPlayerPhase};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake1);
}

TEST_F(PerfectPlayerTest, chance_phase_throw_error) {
  State state{4, stochastic_nim::kPlayer0, stochastic_nim::kPlayer1, stochastic_nim::kChancePhase};
  EXPECT_THROW(get_action_response(state), util::Exception);
}

TEST_F(PerfectPlayerTest, greater_than_starting_stones_throw_error) {
  State state{999, stochastic_nim::kPlayer0, stochastic_nim::kPlayer1,
              stochastic_nim::kPlayerPhase};

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
  Rules::apply(state, Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase));

  EXPECT_EQ(state.stones_left, 18);
  EXPECT_EQ(Rules::get_current_player(state), 0);
}

TEST(StochasticNimGameTest, VerifyChanceStatus) {
  State state;
  Rules::init_state(state);

  Rules::apply(state, Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase));
  if (stochastic_nim::kChanceDistributionSize == 0) {
    EXPECT_EQ(state.current_phase, stochastic_nim::kPlayerPhase);
    EXPECT_EQ(Rules::get_game_phase(state), 0);
    EXPECT_FALSE(Rules::is_chance_phase(Rules::get_game_phase(state)));
  } else {
    EXPECT_EQ(state.current_phase, stochastic_nim::kChancePhase);
    EXPECT_EQ(Rules::get_game_phase(state), 1);
    EXPECT_TRUE(Rules::is_chance_phase(Rules::get_game_phase(state)));
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

  Rules::apply(state, Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase));

  auto dist = Rules::get_chance_distribution(state);

  EXPECT_NEAR(dist.get(Move(0, stochastic_nim::kChancePhase)), 0.2, 1e-6);
  EXPECT_NEAR(dist.get(Move(1, stochastic_nim::kChancePhase)), 0.3, 1e-6);
  EXPECT_NEAR(dist.get(Move(2, stochastic_nim::kChancePhase)), 0.5, 1e-6);
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

    Rules::apply(state, Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase));

    auto dist = Rules::get_chance_distribution(state);
    Move chance_move = dist.sample(util::Random::default_prng());
    Rules::apply(state, chance_move);

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
  std::vector<Move> moves = {Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase)};

  for (Move move : moves) {
    Rules::apply(state, move);
    Rules::apply(state, Move(0, stochastic_nim::kChancePhase));
  }

  auto result = Rules::analyze(state);
  bool terminal = result.is_terminal();
  GameResults::Tensor outcome = result.outcome();

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[0], 1);
}

TEST(StochasticNimGameTest, Player1Wins) {
  State state;
  Rules::init_state(state);
  std::vector<Move> moves = {Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake1, stochastic_nim::kPlayerPhase),
                             Move(stochastic_nim::kTake2, stochastic_nim::kPlayerPhase)};

  for (Move move : moves) {
    Rules::apply(state, move);
    // chance action
    Rules::apply(state, Move(0, stochastic_nim::kChancePhase));
  }

  GameResults::Tensor outcome;
  auto result = Rules::analyze(state);
  bool terminal = result.is_terminal();
  outcome = result.outcome();

  EXPECT_TRUE(terminal);
  EXPECT_EQ(outcome[1], 1);
}

TEST(StochasticNimGameTest, InvalidMove) {
  State state;
  Rules::init_state(state);
  EXPECT_THROW(Rules::apply(state, Move(-1, stochastic_nim::kPlayerPhase)), std::invalid_argument);
  EXPECT_THROW(Rules::apply(state, Move(3, stochastic_nim::kPlayerPhase)), std::invalid_argument);
}

TEST(StochasticNimGameTest, MoveProbMass) {
  State state;
  state.stones_left = 1;
  state.current_player = 0;
  state.current_phase = stochastic_nim::kChancePhase;
  ChanceDistribution dist = Rules::get_chance_distribution(state);

  EXPECT_NEAR(dist.get(Move(0, stochastic_nim::kChancePhase)), 0.2, 1e-6);
  EXPECT_NEAR(dist.get(Move(1, stochastic_nim::kChancePhase)), 0.8, 1e-6);
  EXPECT_NEAR(dist.get(Move(2, stochastic_nim::kChancePhase)), 0.0, 1e-6);
}

TEST(StochasticNimGameTest, tensorize) {
  State state;
  Rules::init_state(state);
  Rules::apply(state, Move(stochastic_nim::kTake2, stochastic_nim::kPlayerPhase));  // Player 0
  Rules::apply(state, Move(0, stochastic_nim::kChancePhase));                       // chance
  Rules::apply(state, Move(stochastic_nim::kTake1, stochastic_nim::kPlayerPhase));  // Player 1
  Rules::apply(state, Move(0, stochastic_nim::kChancePhase));                       // chance

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
