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

using Game = nim::Game;
using State = Game::State;

class ManagerTest : public testing::Test {
 protected:

  using Manager = mcts::Manager<Game>;
  using ManagerParams = mcts::ManagerParams<Game>;
  using Node = mcts::Node<Game>;
  using StateHistory = Game::StateHistory;
  using SearchThread = mcts::SearchThread<Game>;
  using action_t = core::action_t;
  using edge_t = mcts::Node<Game>::edge_t;
  using LookupTable = mcts::Node<Game>::LookupTable;
  using node_pool_index_t = mcts::Node<Game>::node_pool_index_t;
  using edge_pool_index_t = mcts::Node<Game>::edge_pool_index_t;
  using ValueArray = Game::Types::ValueArray;

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

    node_pool_index_t add_child_by_action(node_pool_index_t node_ix, action_t action) {
      Node* node = get_node_by_index(node_ix);
      edge_t* edge = node->get_edge(action);
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

    void modify_node_Q(node_pool_index_t index, ValueArray Q) {
      Node* node = get_node_by_index(index);
      node->stats().Q = Q;
      node->stats().Q_sq = Q.cwiseProduct(Q);
    }

    void modify_node_RN(node_pool_index_t index, int RN) {
      Node* node = get_node_by_index(index);
      node->stats().RN = RN;
    }

    std::string print_tree(node_pool_index_t node_ix, const State& prev_state, int num_indent=0) {
      std::ostringstream oss;
      Node* node = get_node_by_index(node_ix);

      if (num_indent == 0) {
        oss << std::string(num_indent * 2, ' ');
      }
      else{
        oss << std::string((num_indent - 1) * 2, ' ') << "|-";
      }
      oss << "Node " << node_ix << ": " << Game::IO::state_repr(prev_state) << " RN = " << node->stats().RN
      << ": Q= " << node->stats().Q.transpose() << std::endl;

      for (int i = 0; i < node->stable_data().num_valid_actions; ++i) {
        edge_t* edge = node->get_edge(i);

        if (edge->child_index == -1) {
          continue;
        }
        edge_pool_index_t edge_index = i + node->get_first_edge_index();

        State new_state = prev_state;
        StateHistory history;
        history.update(new_state);
        Game::Rules::apply(history, edge->action);
        new_state = history.current();

        oss << std::string(num_indent * 2, ' ') << "|-"
        << "Edge " << edge_index
        << ": " << " E = " << edge->N << ", Action = " << edge->action << std::endl;
        if (edge->child_index != -1) {
          oss << print_tree(edge->child_index, new_state, num_indent + 2);
        }
      }
      return oss.str();
    }

    std::string print_tree() {
      State state;
      Game::Rules::init_state(state);
      return print_tree(0, state);
    }

 private:
  ManagerParams manager_params_;
  Manager manager_;
};

TEST_F(ManagerTest, construct_tree) {
  start_manager();
  start_threads();
  search(10);
  EXPECT_EQ(print_tree(),
  "Node 0: [21, 0] RN = 11: Q= 0.5 0.5\n"
  "|-Edge 0:  E = 4, Action = 0\n"
  "  |-Node 1: [20, 1] RN = 4: Q= 0.5 0.5\n"
  "    |-Edge 3:  E = 1, Action = 0\n"
  "      |-Node 4: [19, 0] RN = 1: Q= 0.5 0.5\n"
  "    |-Edge 4:  E = 1, Action = 1\n"
  "      |-Node 5: [18, 0] RN = 1: Q= 0.5 0.5\n"
  "    |-Edge 5:  E = 1, Action = 2\n"
  "      |-Node 6: [17, 0] RN = 1: Q= 0.5 0.5\n"
  "|-Edge 1:  E = 3, Action = 1\n"
  "  |-Node 2: [19, 1] RN = 3: Q= 0.5 0.5\n"
  "    |-Edge 6:  E = 1, Action = 0\n"
  "      |-Node 5: [18, 0] RN = 1: Q= 0.5 0.5\n"
  "    |-Edge 7:  E = 1, Action = 1\n"
  "      |-Node 6: [17, 0] RN = 1: Q= 0.5 0.5\n"
  "|-Edge 2:  E = 3, Action = 2\n"
  "  |-Node 3: [18, 1] RN = 3: Q= 0.5 0.5\n"
  "    |-Edge 9:  E = 1, Action = 0\n"
  "      |-Node 6: [17, 0] RN = 1: Q= 0.5 0.5\n"
  "    |-Edge 10:  E = 1, Action = 1\n"
  "      |-Node 7: [16, 0] RN = 1: Q= 0.5 0.5\n");
}

TEST_F(ManagerTest, backprop) {
  start_manager();
  start_threads();
  search(0);

  init_search_thread();
  node_pool_index_t child_index1 = add_child_by_action(0, 1);
  node_pool_index_t child_index2 = add_child_by_action(child_index1, 2);
  node_pool_index_t child_index3 = add_child_by_action(child_index2, 0);
  ValueArray v;
  v << 0.8, 0.2;
  modify_node_Q(child_index2, v);
  modify_node_RN(child_index2, 100);
  v << 0.1, 0.9;
  modify_node_Q(child_index3, v);
  modify_node_RN(child_index3, 9999);

  EXPECT_EQ(print_tree(),
  "Node 0: [21, 0] RN = 1: Q= 0.5 0.5\n"
  "|-Edge 1:  E = 0, Action = 1\n"
  "  |-Node 1: [19, 1] RN = 0: Q= 0.5 0.5\n"
  "    |-Edge 5:  E = 0, Action = 2\n"
  "      |-Node 2: [16, 0] RN = 100: Q= 0.8 0.2\n"
  "        |-Edge 6:  E = 0, Action = 0\n"
  "          |-Node 3: [15, 1] RN = 9999: Q= 0.1 0.9\n");
  }

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
