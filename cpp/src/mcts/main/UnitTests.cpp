#include <core/tests/Common.hpp>
#include <games/nim/Constants.hpp>
#include <games/nim/Game.hpp>
#include <games/connect4/Game.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchThread.hpp>
#include <mcts/SharedData.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>
#include <mcts/Node.hpp>
#include <mcts/Manager.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

class ManagerTest : public testing::Test {
 protected:
  using Game = nim::Game;
  using Manager = mcts::Manager<Game>;
  using ManagerParams = mcts::ManagerParams<Game>;

 public:
  ManagerTest():
    manager_params_(mcts::kCompetitive),
    manager_(manager_params_) {}

 private:
  ManagerParams manager_params_;
  Manager manager_;
};

TEST_F(ManagerTest, backprop) {
  std::cout << "backprop" << std::endl;
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
