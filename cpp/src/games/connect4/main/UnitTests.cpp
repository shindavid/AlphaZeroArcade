#include <core/tests/Common.hpp>
#include <games/connect4/Constants.hpp>
#include <games/connect4/Game.hpp>
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

/*
 * Tests connect4 symmetry classes.
 */

using Game = c4::Game;
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

State make_init_state() {
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, 3);
  Rules::apply(history, 4);
  Rules::apply(history, 3);
  return history.current();
}

PolicyTensor make_policy(int move1, int move2) {
  PolicyTensor tensor;
  tensor.setZero();
  tensor(move1) = 1;
  tensor(move2) = 1;
  return tensor;
}

const std::string init_state_repr =
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | |R| | | |\n"
    "| | | |R|Y| | |\n";

std::string get_repr(const State& state) {
  std::ostringstream ss;
  IO::print_state(ss, state);

  std::string s = ss.str();
  std::vector<std::string> lines;
  std::istringstream iss(s);
  std::string line;
  while (std::getline(iss, line)) {
    lines.push_back(line);
  }

  std::string repr;
  for (int i = 0; i < 6; ++i) {
    repr += lines[i];
    repr += '\n';
  }

  return repr;
}

TEST(Symmetry, identity) {
  State state = make_init_state();

  group::element_t sym = groups::D1::kIdentity;
  group::element_t inv_sym = groups::D1::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr = init_state_repr;

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(0, 1);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, flip) {
  State state = make_init_state();

  group::element_t sym = groups::D1::kFlip;
  group::element_t inv_sym = groups::D1::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
      "| | | | | | | |\n"
      "| | | | | | | |\n"
      "| | | | | | | |\n"
      "| | | | | | | |\n"
      "| | | |R| | | |\n"
      "| | |Y|R| | | |\n";

  EXPECT_EQ(repr, expected_repr);
  Game::Symmetries::apply(state, inv_sym);
  EXPECT_EQ(get_repr(state), init_state_repr);

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(5, 6);
  EXPECT_TRUE(eigen_util::equal(policy, expected_policy));
  Game::Symmetries::apply(policy, inv_sym);
  EXPECT_TRUE(eigen_util::equal(policy, init_policy));
}

TEST(Symmetry, action_transforms) {
  core::tests::Common<Game>::gtest_action_transforms();
}

class SearchThreadTest : public testing::Test {
 protected:
  using ManagerParams = mcts::ManagerParams<Game>;
  using NNEvaluationService = mcts::NNEvaluationService<Game>;
  using SearchParams = mcts::SearchParams;
  using SearchThread = mcts::SearchThread<Game>;
  using SharedData = mcts::SharedData<Game>;
  using Node = mcts::Node<Game>;

  SearchThreadTest()
      : manager_id_(0),
        thread_id_(0),
        manager_params_(mcts::kCompetitive),
        shared_data_(manager_params_, manager_id_),
        nn_eval_service_(nullptr),
        search_thread_(&shared_data_, nn_eval_service_, &manager_params_, thread_id_) {}

  void TearDown() override {
    shared_data_.shutting_down = true;  // needed for SearchThread loop to exit
  }

  void init(int tree_size_limit, bool full_search, bool ponder=false) {
    SearchParams search_params{tree_size_limit, full_search, ponder};
    shared_data_.search_params = search_params;
    shared_data_.clear();

    bool add_noise = search_params.full_search && manager_params_.dirichlet_mult > 0;
    shared_data_.init_root_info(add_noise);
  }

  void test_init_root_node() {
    Node* root = search_thread_.init_root_node();
    EXPECT_NE(root, nullptr);

    Node::stable_data_t stable_data = root->stable_data();
    EXPECT_EQ(stable_data.valid_action_mask.count(), 7);
    EXPECT_EQ(stable_data.num_valid_actions, 7);
    EXPECT_EQ(stable_data.current_player, 0);
    EXPECT_FALSE(stable_data.terminal);
    EXPECT_TRUE(stable_data.VT_valid);
    EXPECT_NEAR(stable_data.VT(0), 1.0 / 3, 1e-6);
    EXPECT_NEAR(stable_data.VT(1), 1.0 / 3, 1e-6);
    EXPECT_NEAR(stable_data.VT(2), 1.0 / 3, 1e-6);

    Node::stats_t stats = root->stats();
    EXPECT_EQ(stats.RN, 1);
    EXPECT_EQ(stats.VN, 0);
    EXPECT_EQ(stats.Q(0), 0.5);
    EXPECT_EQ(stats.Q(1), 0.5);

    EXPECT_TRUE(root->edges_initialized());
    EXPECT_FALSE(root->trivial());
  }

  void perform_visits() { search_thread_.perform_visits(); }

  int manager_id_;
  int thread_id_;
  ManagerParams manager_params_;
  SharedData shared_data_;
  NNEvaluationService* nn_eval_service_;
  SearchThread search_thread_;
};

TEST_F(SearchThreadTest, init_root_node) {
  init(1, true);
  test_init_root_node();
}

TEST_F(SearchThreadTest, something_else) {
  init(1, true);
  perform_visits();
  // TODO test something
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
