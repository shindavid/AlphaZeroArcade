#include <core/concepts/Game.hpp>
#include <generic_players/MctsPlayer.hpp>
#include <mcts/Manager.hpp>

#include <gtest/gtest.h>

template<core::concepts::Game Game>
class MctsPlayerTest : public ::testing::Test {
 protected:

  using Manager = mcts::Manager<Game>;
  using ManagerParams = mcts::ManagerParams<Game>;
  using MctsPlayer = generic::MctsPlayer<Game>;
  using MctsPlayerParams = MctsPlayer::Params;


public:
  MctsPlayerTest() : manager_params_(create_manager_params()), player_params_(mcts::kCompetitive) {}

  ManagerParams create_manager_params() {
    ManagerParams params(mcts::kCompetitive);
    params.no_model = true;
    return params;
  }

  void SetUp() override {
    mcts_manager_ = new Manager(manager_params_);
    mcts_player_ = new MctsPlayer(player_params_, mcts_manager_);
  }

  void TearDown() override {
    delete mcts_manager_;
    delete mcts_player_;
  }

private:
  ManagerParams manager_params_;
  MctsPlayerParams player_params_;
  Manager* mcts_manager_;
  MctsPlayer* mcts_player_;
};

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}