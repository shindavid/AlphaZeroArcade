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

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

using Game = c4::Game;
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;
using search_path_t = mcts::SearchThread<Game>::search_path_t_public;
using edge_t = mcts::Node<Game>::edge_t;
using node_pool_index_t = mcts::Node<Game>::node_pool_index_t;
using SearchParams = mcts::SearchParams;


class BackpropTest : public testing::Test {
 protected:
  using Node = mcts::Node<Game>;
  using SharedData = mcts::SharedData<Game>;

  BackpropTest() :
        manager_id_(0),
        manager_params_(mcts::kCompetitive),
        shared_data_(manager_params_, manager_id_),
        root_(nullptr) {}

  void test_backprop() {
    SearchParams search_params{1, true, false};
    shared_data_.search_params = search_params;
    shared_data_.init_root_info(false);
    node_pool_index_t root_index = shared_data_.root_info.node_index;
    root_ = shared_data_.lookup_table.get_node(root_index);

    auto child1_index = shared_data_.lookup_table.alloc_node();
    Node* child1 = shared_data_.lookup_table.get_node(child1_index);

    root_->initialize_edges();
    root_->get_edge(0)->child_index = child1_index;
    root_->get_edge(0)->action = 1;

    child1->stats().Q(0) = 0.6;
    child1->stats().Q(1) = 0.4;
    child1->stats().RN = 0;

    search_path_t search_path;
    search_path.emplace_back(root_, nullptr);
    search_path.emplace_back(child1, root_->get_edge(0));

    Node* last_node = search_path_.back().node;
    const auto& stable_data = last_node->stable_data();
    auto value = Game::GameResults::to_value_array(stable_data.VT);

    last_node->update_stats([&] {
      last_node->stats().init_q(value, true);
      last_node->stats().RN++;
    });

    for (int i = search_path_.size() - 2; i >= 0; --i) {
      edge_t* edge = search_path_[i].edge;
      Node* node = search_path_[i].node;

      // NOTE: always update the edge first, then the parent node
      node->update_stats([&] {
        edge->N++;
        node->stats().RN++;
      });
    }

    // std::cout << "Root stats: " << root_->stats().Q << std::endl;
  }

  int manager_id_;
  mcts::ManagerParams<Game> manager_params_ = mcts::kCompetitive;
  SharedData shared_data_;
  Node* root_;
  search_path_t search_path_;
};

TEST_F(BackpropTest, backprop) {
  test_backprop();
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
