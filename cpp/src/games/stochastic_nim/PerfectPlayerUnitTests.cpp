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
  using Params = PerfectPlayer::Params;

 public:
  PerfectPlayerTest() : player_(PerfectPlayer(Params())) {}
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

 private:
  PerfectPlayer player_;
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

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

