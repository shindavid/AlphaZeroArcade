#include <core/tests/Common.hpp>
#include <games/nim/Constants.hpp>
#include <games/nim/Game.hpp>
#include <games/connect4/Game.hpp>
#include <mcts/Manager.hpp>
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

    Node* get_node_by_index(node_pool_index_t index) {
      return manager_.shared_data_.lookup_table.get_node(index);
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

    /*
     * This function prints the tree structure starting from root.
     * It recursively traverses the tree and prints each node and its edges.
     * The format is as follows:
     * - Each node is printed with its index, state representation, RN (visit count), and Q values.
     * - Each edge is printed with its index, visit count (E), and action.
     * - Indentation is used to represent the tree structure, with each level of depth indented further.
    */
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

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
