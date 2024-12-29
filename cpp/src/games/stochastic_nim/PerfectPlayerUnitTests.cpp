#include <games/stochastic_nim/PerfectPlayer.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/RepoUtil.hpp>

#include <gtest/gtest.h>

#include <sstream>

class PerfectPlayerTest : public testing::Test {
 protected:
  using PerfectPlayer = stochastic_nim::PerfectPlayer;
  using State = PerfectPlayer::State;
  using Types = PerfectPlayer::Types;

 public:
  PerfectPlayerTest() : player_(PerfectPlayer()) {}
  void test_tensor_values(const std::string& testname) {

    std::ostringstream oss;
    oss << player_.get_state_action_tensor();

    boost::filesystem::path base_dir = util::Repo::root() / "goldenfiles" / "perfect_player";
    boost::filesystem::path file_path = base_dir / (testname + ".txt");
    if (IS_MACRO_ENABLED(WRITE_GOLDENFILES)) {
      boost_util::write_str_to_file(oss.str(), file_path);
    }

    std::ifstream golden_file(file_path);
    std::string expected_string((std::istreambuf_iterator<char>(golden_file)), std::istreambuf_iterator<char>());

    EXPECT_EQ(oss.str(), expected_string);
  }

  core::action_t get_action_response(const State& state) {
    Types::ActionMask valid_actions;
    return player_.get_action_response(state, valid_actions).action;
  }

  float get_state_action_value(const State& state, core::action_t action) {
    return player_.get_state_action_value(state, action);
  }
 private:
  PerfectPlayer player_;
};

class PerfectStrategyTest: public testing::Test {
 protected:
  using PerfectStrategy = stochastic_nim::PerfectStrategy;
  using Params = PerfectStrategy::Params;

 public:
  PerfectStrategy get_strategy(int starting_stones, int max_stones_to_take, const float* dist, int dist_size) {
    Params params{starting_stones, max_stones_to_take, dist, dist_size};
    return PerfectStrategy(params);
  }
};

TEST_F(PerfectPlayerTest, tensor_values) {
  test_tensor_values("stochastic_nim_tensor_values");
}

TEST_F(PerfectPlayerTest, 4_stones_player0) {
  State state{4, PerfectPlayer::Player0, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 5_stones_player0) {
  State state{5, PerfectPlayer::Player0, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 6_stones_player0) {
  State state{6, PerfectPlayer::Player0, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake1);
}

TEST_F(PerfectPlayerTest, 4_stones_player1) {
  State state{4, PerfectPlayer::Player1, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 5_stones_player1) {
  State state{5, PerfectPlayer::Player1, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake3);
}

TEST_F(PerfectPlayerTest, 6_stones_player1) {
  State state{6, PerfectPlayer::Player1, stochastic_nim::kPlayerMode};
  EXPECT_EQ(get_action_response(state), stochastic_nim::kTake1);
}

TEST_F(PerfectPlayerTest, chance_mode_throw_error) {
  State state{4, PerfectPlayer::Player0, stochastic_nim::kChanceMode};
  EXPECT_THROW(get_action_response(state), std::invalid_argument);
}

TEST_F(PerfectPlayerTest, Q_value_4_stones_left) {
  State state{4, PerfectPlayer::Player0, stochastic_nim::kPlayerMode};
  EXPECT_NEAR(get_state_action_value(state, stochastic_nim::kTake3), 0.8, 1e-6);
  EXPECT_NEAR(get_state_action_value(state, stochastic_nim::kTake2), 0.5, 1e-6);
  EXPECT_NEAR(get_state_action_value(state, stochastic_nim::kTake1), 0.0, 1e-6);
}

TEST_F(PerfectPlayerTest, Q_value_3_stones_left) {
  State state{3, PerfectPlayer::Player0, stochastic_nim::kPlayerMode};
  EXPECT_NEAR(get_state_action_value(state, stochastic_nim::kTake3), 1.0, 1e-6);
  EXPECT_NEAR(get_state_action_value(state, stochastic_nim::kTake2), 0.8, 1e-6);
  EXPECT_NEAR(get_state_action_value(state, stochastic_nim::kTake1), 0.5, 1e-6);
}

TEST_F(PerfectPlayerTest, Q_value_2_stones_left) {
  State state{2, PerfectPlayer::Player0, stochastic_nim::kPlayerMode};
  EXPECT_NEAR(get_state_action_value(state, stochastic_nim::kTake2), 1.0, 1e-6);
  EXPECT_NEAR(get_state_action_value(state, stochastic_nim::kTake1), 0.8, 1e-6);
}

TEST_F(PerfectPlayerTest, Q_value_1_stones_left) {
  State state{1, PerfectPlayer::Player0, stochastic_nim::kPlayerMode};
  EXPECT_NEAR(get_state_action_value(state, stochastic_nim::kTake1), 1.0, 1e-6);
}

TEST_F(PerfectStrategyTest, 4_stones) {
  PerfectStrategy strategy = get_strategy(4, 3, stochastic_nim::kChanceEventProbs, 3);
  EXPECT_NEAR(strategy.get_state_value()[4], 0.04, 1e-6);
  EXPECT_NEAR(strategy.get_state_value()[3], 0.0, 1e-6);
  EXPECT_NEAR(strategy.get_state_value()[2], 0.5, 1e-6);
  EXPECT_NEAR(strategy.get_state_value()[1], 0.8, 1e-6);
  EXPECT_NEAR(strategy.get_state_value()[0], 1.0, 1e-6);
  EXPECT_EQ(strategy.get_optimal_action()[4], 3);
  EXPECT_EQ(strategy.get_optimal_action()[3], 3);
  EXPECT_EQ(strategy.get_optimal_action()[2], 2);
  EXPECT_EQ(strategy.get_optimal_action()[1], 1);
  EXPECT_EQ(strategy.get_optimal_action()[0], -1);
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

