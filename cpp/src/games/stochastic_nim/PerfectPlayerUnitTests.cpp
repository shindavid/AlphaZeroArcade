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

 private:
  PerfectPlayer player_;
};

TEST_F(PerfectPlayerTest, tensor_values) {
  test_tensor_values("stochastic_nim_tensor_values");
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

