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
  using Node = mcts::Node<Game>;
  using StateHistory = Game::StateHistory;
  using SearchThread = mcts::SearchThread<Game>;
  using action_t = core::action_t;
  using edge_t = mcts::Node<Game>::edge_t;
  using LookupTable = mcts::Node<Game>::LookupTable;
  using node_pool_index_t = mcts::Node<Game>::node_pool_index_t;

 public:
  ManagerTest():
    manager_params_(create_manager_params()),
    manager_(manager_params_) {}

    static ManagerParams create_manager_params() {
      ManagerParams params(mcts::kCompetitive);
      params.no_model = true;
      return params;
    }

    void start_manager() {
      manager_.start();
    }

    void start_threads() {
      manager_.start_threads();
    }

    void search(int num_searches = 0){
      mcts::SearchParams search_params(num_searches, true);
      manager_.search(search_params);
    }

    Node* get_root_node() {
      return manager_.shared_data_.get_root_node();
    }

    Node* get_node_by_index(node_pool_index_t index) {
      return manager_.shared_data_.lookup_table.get_node(index);
    }

    SearchThread* get_search_thread() {
      return manager_.search_threads_[0];
    }

    void init_search_thread() {
      SearchThread* search_thread = get_search_thread();
      search_thread->search_path_.clear();
      search_thread->search_path_.emplace_back(get_root_node());
    }

    LookupTable& lookup_table() {
      return manager_.shared_data_.lookup_table;
    }

    StateHistory& get_raw_history() {
      return get_search_thread()->raw_history_;
    }

    node_pool_index_t add_child_by_action(Node* node, action_t action) {

      edge_t* edge = node->get_edge(node->get_first_edge_index() + action);
      if (edge->action == action && edge->child_index != -1) {
        throw std::invalid_argument("Child already exists");
      }

      Game::Rules::apply(get_raw_history(), edge->action);

      edge->child_index = lookup_table().alloc_node();
      Node* child = lookup_table().get_node(edge->child_index);
      new (child) Node(&lookup_table(), get_raw_history());

      child->initialize_edges();
      get_search_thread()->init_node(&get_raw_history(), edge->child_index, child);

      return edge->child_index;
    }


 private:
  ManagerParams manager_params_;
  Manager manager_;
};

TEST_F(ManagerTest, backprop) {
  start_manager();
  start_threads();
  search(0);

  init_search_thread();
  node_pool_index_t child_index1 = add_child_by_action(get_node_by_index(0), 1);
  node_pool_index_t child_index2 = add_child_by_action(get_node_by_index(0), 2);
  node_pool_index_t child_index3 = add_child_by_action(get_node_by_index(1), 0);
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
