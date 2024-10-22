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
    manager_params_(create_manager_params()),
    manager_(manager_params_) {}

    static ManagerParams create_manager_params() {
      ManagerParams params(mcts::kCompetitive);
      params.no_model = true;
      return params;
    }

    void start_threads() {
      manager_.start_threads();
    }

 private:
  ManagerParams manager_params_;
  Manager manager_;
};

TEST_F(ManagerTest, backprop) {
  std::cout << "====TEST: backprop" << std::endl;
  start_threads();
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
