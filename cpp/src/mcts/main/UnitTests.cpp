#include <core/tests/Common.hpp>
#include <games/nim/Game.hpp>
#include <games/tictactoe/Game.hpp>
#include <mcts/Graph.hpp>
#include <mcts/Manager.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationServiceBase.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchThread.hpp>
#include <mcts/SharedData.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

class MockNNEvaluationService : public mcts::NNEvaluationServiceBase<nim::Game> {
 public:
  using NNEvaluation = mcts::NNEvaluation<nim::Game>;
  using ValueTensor = NNEvaluation::ValueTensor;
  using PolicyTensor = NNEvaluation::PolicyTensor;
  using ActionValueTensor = NNEvaluation::ActionValueTensor;
  using ActionMask = NNEvaluation::ActionMask;

  MockNNEvaluationService(bool smart) : smart_(smart) {}

  void evaluate(const NNEvaluationRequest& request) override {
    ValueTensor value;
    PolicyTensor policy;
    ActionValueTensor action_values;
    group::element_t sym = group::kIdentity;

    for (NNEvaluationRequest::Item& item : request.items()) {
      ActionMask valid_actions = item.node()->stable_data().valid_action_mask;
      core::seat_index_t cp = item.node()->stable_data().current_player;

      const nim::Game::State& state = item.cur_state();

      bool winning = state.stones_left % (1 + nim::kMaxStonesToTake) != 0;
      if (winning) {
        core::action_t winning_move = (state.stones_left) % (1 + nim::kMaxStonesToTake) - 1;

        // these are logits
        float winning_v = smart_ ? 2 : 0;
        float losing_v = smart_ ? 0 : 2;

        float winning_action_p = smart_ ? 2 : 0;
        float losing_action_p = smart_ ? 0 : 2;

        value.setValues({winning_v, losing_v});

        policy.setConstant(losing_action_p);
        policy(winning_move) = winning_action_p;

        action_values.setConstant(losing_v);
        action_values(winning_move) = winning_v;
      } else {
        value.setZero();
        policy.setZero();
        action_values.setZero();
      }

      item.set_eval(
          std::make_shared<NNEvaluation>(value, policy, action_values, valid_actions, sym, cp));
    }
  }

 private:
  bool smart_;
};

template<core::concepts::Game Game>
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
  using Service = mcts::NNEvaluationServiceBase<Game>;
  using State = Game::State;

 public:
  ManagerTest() : manager_params_(create_manager_params()) {}

  ~ManagerTest() override {
    auto graph_viz = manager_params_.get_graph_viz();
    delete graph_viz;
    // delete manager_;
  }

  static ManagerParams create_manager_params() {
    mcts::GraphViz<Game>* graph_viz = new mcts::GraphViz<Game>();
    ManagerParams params(mcts::kCompetitive, graph_viz);
    params.no_model = true;
    return params;
  }

  void init_manager(Service* service = nullptr) {
    manager_ = new Manager(manager_params_, service);
  }

  void start_manager(const std::vector<core::action_t>& initial_actions = {}) {
    manager_->start();
    for (core::action_t action : initial_actions) {
      manager_->shared_data()->update_state(action);
    }
    this->initial_actions_ = initial_actions;
  }

  ManagerParams& manager_params() { return manager_params_; }

  void start_threads() { manager_->start_threads(); }

  void search(int num_searches = 0) {
    mcts::SearchParams search_params(num_searches, true);
    manager_->search(search_params);
  }

  Node* get_node_by_index(node_pool_index_t index) {
    return manager_->shared_data()->lookup_table.get_node(index);
  }

  std::string print_tree(node_pool_index_t node_ix, const State& prev_state, int num_indent = 0) {
    std::ostringstream oss;
    Node* node = get_node_by_index(node_ix);

    if (num_indent == 0) {
      oss << std::string(num_indent * 2, ' ');
    } else {
      const char* marker = node->is_terminal() ? "|*" : "|-";
      oss << std::string((num_indent - 1) * 2, ' ') << marker;
    }
    oss << "Node " << node_ix << ": " << Game::IO::compact_state_repr(prev_state)
        << " RN = " << node->stats().RN << ": Q = " << node->stats().Q.transpose() << std::endl;

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
          << "Edge " << edge_index << ": " << " E = " << edge->E << ", Action = " << edge->action
          << std::endl;
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
   * - Indentation is used to represent the tree structure, with each level of depth indented
   * further.
   */
  using Rules = Game::Rules;
  std::string print_tree() {
    StateHistory history;
    history.initialize(Rules{});
    for (core::action_t action : initial_actions_) {
      Game::Rules::apply(history, action);
    }
    return print_tree(0, history.current());
  }

  ManagerParams& get_manager_params() { return manager_params_; }

 private:
  ManagerParams manager_params_;
  Manager* manager_ = nullptr;
  std::vector<core::action_t> initial_actions_;
};

using NimManagerTest = ManagerTest<nim::Game>;
TEST_F(NimManagerTest, uniform_search) {
  init_manager();
  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake2};
  start_manager(initial_actions);
  start_threads();
  search(10);
  std::string tree_str = print_tree();
  EXPECT_EQ(tree_str,
            "Node 0: [4, 0] RN = 11: Q = 0.318182 0.681818\n"
            "|-Edge 0:  E = 5, Action = 0\n"
            "  |-Node 1: [3, 1] RN = 5: Q = 0.3 0.7\n"
            "    |-Edge 3:  E = 1, Action = 0\n"
            "      |-Node 4: [2, 0] RN = 1: Q = 0.5 0.5\n"
            "    |-Edge 4:  E = 1, Action = 1\n"
            "      |-Node 5: [1, 0] RN = 1: Q = 0.5 0.5\n"
            "    |-Edge 5:  E = 2, Action = 2\n"
            "      |*Node 6: [0, 0] RN = 2: Q = 0 1\n"
            "|-Edge 1:  E = 3, Action = 1\n"
            "  |-Node 2: [2, 1] RN = 3: Q = 0.333333 0.666667\n"
            "    |-Edge 6:  E = 1, Action = 0\n"
            "      |-Node 5: [1, 0] RN = 1: Q = 0.5 0.5\n"
            "    |-Edge 7:  E = 1, Action = 1\n"
            "      |*Node 6: [0, 0] RN = 2: Q = 0 1\n"
            "|-Edge 2:  E = 2, Action = 2\n"
            "  |-Node 3: [1, 1] RN = 2: Q = 0.25 0.75\n"
            "    |-Edge 8:  E = 1, Action = 0\n"
            "      |*Node 6: [0, 0] RN = 2: Q = 0 1\n");
}

TEST_F(NimManagerTest, smart_search) {
  MockNNEvaluationService mock_service(true);
  init_manager(&mock_service);
  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake2};

  start_manager(initial_actions);
  start_threads();
  search(10);
  std::string tree_str = print_tree();
  EXPECT_EQ(tree_str,
            "Node 0: [4, 0] RN = 11: Q = 0.0779644  0.922036\n"
            "|-Edge 0:  E = 4, Action = 0\n"
            "  |-Node 1: [3, 1] RN = 4: Q = 0.0298007  0.970199\n"
            "    |-Edge 5:  E = 3, Action = 2\n"
            "      |*Node 4: [0, 0] RN = 3: Q = 0 1\n"
            "|-Edge 1:  E = 3, Action = 1\n"
            "  |-Node 2: [2, 1] RN = 3: Q = 0.0397343  0.960266\n"
            "    |-Edge 7:  E = 2, Action = 1\n"
            "      |*Node 4: [0, 0] RN = 3: Q = 0 1\n"
            "|-Edge 2:  E = 3, Action = 2\n"
            "  |-Node 3: [1, 1] RN = 3: Q = 0.0397343  0.960266\n"
            "    |-Edge 8:  E = 2, Action = 0\n"
            "      |*Node 4: [0, 0] RN = 3: Q = 0 1\n");
}

TEST_F(NimManagerTest, dumb_search) {
  MockNNEvaluationService mock_service(false);
  init_manager(&mock_service);
  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake2};

  start_manager(initial_actions);
  start_threads();
  search(10);
  std::string tree_str = print_tree();
  EXPECT_EQ(tree_str,
            "Node 0: [4, 0] RN = 11: Q = 0.489163 0.510837\n"
            "|-Edge 0:  E = 4, Action = 0\n"
            "  |-Node 1: [3, 1] RN = 4: Q = 0.5 0.5\n"
            "    |-Edge 3:  E = 2, Action = 0\n"
            "      |-Node 2: [2, 0] RN = 2: Q = 0.279801 0.720199\n"
            "        |-Edge 6:  E = 1, Action = 0\n"
            "          |-Node 5: [1, 1] RN = 2: Q = 0.440399 0.559601\n"
            "            |-Edge 11:  E = 1, Action = 0\n"
            "              |*Node 6: [0, 0] RN = 1: Q = 0 1\n"
            "    |-Edge 4:  E = 1, Action = 1\n"
            "      |-Node 4: [1, 0] RN = 2: Q = 0.559601 0.440399\n"
            "        |-Edge 10:  E = 1, Action = 0\n"
            "          |*Node 7: [0, 1] RN = 1: Q = 1 0\n"
            "|-Edge 1:  E = 4, Action = 1\n"
            "  |-Node 3: [2, 1] RN = 4: Q = 0.5 0.5\n"
            "    |-Edge 8:  E = 2, Action = 0\n"
            "      |-Node 4: [1, 0] RN = 2: Q = 0.559601 0.440399\n"
            "        |-Edge 10:  E = 1, Action = 0\n"
            "          |*Node 7: [0, 1] RN = 1: Q = 1 0\n"
            "    |-Edge 9:  E = 1, Action = 1\n"
            "      |*Node 6: [0, 0] RN = 1: Q = 0 1\n"
            "|-Edge 2:  E = 2, Action = 2\n"
            "  |-Node 5: [1, 1] RN = 2: Q = 0.440399 0.559601\n"
            "    |-Edge 11:  E = 1, Action = 0\n"
            "      |*Node 6: [0, 0] RN = 1: Q = 0 1\n");
}
#ifdef STORE_STATES
TEST_F(NimManagerTest, graph_viz) {
  init_manager();
  start_manager();
  start_threads();
  search(20);

  // std::string file_path = "./py/alphazero/dashboard/Graph/graph_jsons/nim_uniform.json";
  boost::filesystem::path file_path =
      util::Repo::root() / "goldenfiles" / "mcts_tests" / "nim_uniform.json";
  std::ifstream file(file_path);
  std::string expected_json((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  EXPECT_EQ(get_manager_params().get_graph_viz()->combine_json(), expected_json);
}

TEST_F(NimManagerTest, uniform_search_viz) {
  init_manager();
  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake3, nim::kTake2};
  start_manager(initial_actions);
  start_threads();
  search(100);

  boost::filesystem::path file_path =
      util::Repo::root() / "goldenfiles" / "mcts_tests" / "nim_uniform_4_stones.json";
  std::ifstream file(file_path);
  std::string expected_json((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  EXPECT_EQ(get_manager_params().get_graph_viz()->combine_json(), expected_json);
}

using TicTacToeManagerTest = ManagerTest<tictactoe::Game>;
TEST_F(TicTacToeManagerTest, uniform_search_viz) {
  init_manager();
  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  start_manager(initial_actions);
  start_threads();
  search(100);

  boost::filesystem::path file_path =
      util::Repo::root() / "goldenfiles" / "mcts_tests" / "tictactoe_uniform.json";
  std::ifstream file(file_path);
  std::string expected_json((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  EXPECT_EQ(get_manager_params().get_graph_viz()->combine_json(), expected_json);
}
#endif

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}