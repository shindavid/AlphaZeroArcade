#include <core/tests/Common.hpp>
#include <games/nim/Constants.hpp>
#include <games/nim/Game.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchThread.hpp>
#include <mcts/SharedData.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

using Game = nim::Game;
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;


class BackpropTest : public testing::Test {
 protected:
  using Node = mcts::Node<nim::Game>;
  using SharedData = mcts::SharedData<nim::Game>;

  void SetUp() override {
    shared_data_ = std::make_unique<SharedData>(manager_params_, manager_id_);
  }

  void create_tree() {
    root_ = shared_data_->lookup_table.create_node();
    child1_ = shared_data_->lookup_table.create_node();
    child2_ = shared_data_->lookup_table.create_node();

    root_->add_child(0, child1_);
    root_->add_child(1, child2_);

    root_->stats().Q(0) = 0.5;
    root_->stats().Q(1) = 0.5;
    root_->stats().RN = 10;

    child1_->stats().Q(0) = 0.6;
    child1_->stats().Q(1) = 0.4;
    child1_->stats().RN = 5;

    child2_->stats().Q(0) = 0.4;
    child2_->stats().Q(1) = 0.6;
    child2_->stats().RN = 5;
  }

  void test_backprop() {
    create_tree();
    root_->backprop(1.0, 0);
    EXPECT_EQ(root_->stats().Q(0), 0.55);
    EXPECT_EQ(root_->stats().Q(1), 0.55);
    EXPECT_EQ(root_->stats().RN, 11);
  }

  int manager_id_ = 0;
  mcts::ManagerParams<nim::Game> manager_params_ = mcts::kCompetitive;
  std::unique_ptr<SharedData> shared_data_;
  Node* root_;
  Node* child1_;
  Node* child2_;
};

TEST_F(BackpropTest, backprop) {
  test_backprop();
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
