#include <core/tests/Common.hpp>
#include <games/GameTransforms.hpp>
#include <games/nim/Game.hpp>
#include <games/tictactoe/Game.hpp>
#include <mcts/SearchLog.hpp>
#include <mcts/Manager.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationServiceBase.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchThread.hpp>
#include <mcts/SharedData.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using Nim = game_transform::AddStateStorage<nim::Game>;
using TicTacToe = game_transform::AddStateStorage<tictactoe::Game>;


class MockNNEvaluationService : public mcts::NNEvaluationServiceBase<Nim> {
 public:
  using NNEvaluation = mcts::NNEvaluation<Nim>;
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

      const Nim::State& state = item.cur_state();

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
  using SearchResult = Game::Types::SearchResults;

  static_assert(Node::kStoreStates, "state-storage required for search-log tests");

 public:
  ManagerTest() : manager_params_(create_manager_params()) {}

  ~ManagerTest() override {
    // delete manager_;
  }

  static ManagerParams create_manager_params() {
    ManagerParams params(mcts::kCompetitive);
    params.no_model = true;
    return params;
  }

  void init_manager(Service* service = nullptr) {
    manager_ = new Manager(manager_params_, service);
    const mcts::SharedData<Game>* shared_data = manager_->shared_data();
    search_log_ = new mcts::SearchLog<Game>(shared_data);
    manager_->set_post_visit_func([&] { search_log_->update(); });
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

  const SearchResult* search(int num_searches = 0) {
    mcts::SearchParams search_params(num_searches, true);
    return manager_->search(search_params);
  }

  Node* get_node_by_index(node_pool_index_t index) {
    return manager_->shared_data()->lookup_table.get_node(index);
  }

  mcts::SearchLog<Game>* get_search_log() { return search_log_; }
  ManagerParams& get_manager_params() { return manager_params_; }

  void test_search(const std::string& testname, int num_search,
                   const std::vector<core::action_t>& initial_actions, Service* service) {
    init_manager(service);
    start_manager(initial_actions);
    start_threads();
    const SearchResult* result = search(num_search);

    boost::filesystem::path base_dir = util::Repo::root() / "goldenfiles" / "mcts_tests";

    boost::filesystem::path file_path_result =
        base_dir / (testname + "_result.json");
    boost::filesystem::path file_path_log =
        base_dir / (testname + "_graph.json");

    std::stringstream ss_result;
    boost_util::pretty_print(ss_result, result->to_json());

    if (IS_MACRO_ENABLED(WRITE_GOLDENFILES)) {
      boost_util::write_str_to_file(ss_result.str(), file_path_result);
      boost_util::write_str_to_file(get_search_log()->last_graph_json_str(), file_path_log);
    }

    std::ifstream result_file(file_path_result);
    std::ifstream log_file(file_path_log);

    std::string expected_result_json((std::istreambuf_iterator<char>(result_file)),
                                     std::istreambuf_iterator<char>());
    std::string expected_log_json((std::istreambuf_iterator<char>(log_file)),
                                  std::istreambuf_iterator<char>());

    EXPECT_EQ(ss_result.str(), expected_result_json);
    EXPECT_EQ(get_search_log()->json_str(), expected_log_json);
  }

 private:
  ManagerParams manager_params_;
  Manager* manager_ = nullptr;
  std::vector<core::action_t> initial_actions_;
  mcts::SearchLog<Game>* search_log_ = nullptr;
};

using NimManagerTest = ManagerTest<Nim>;
TEST_F(NimManagerTest, uniform_search) {
  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake2};
  test_search("nim_uniform_10", 10, initial_actions, nullptr);
}

TEST_F(NimManagerTest, smart_search) {
  MockNNEvaluationService mock_service(true);
  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake2};
  test_search("nim_smart_service", 10, initial_actions, &mock_service);
}

TEST_F(NimManagerTest, dumb_search) {
  MockNNEvaluationService mock_service(false);
  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake2};

  test_search("nim_dumb_service", 10, initial_actions, &mock_service);
}

TEST_F(NimManagerTest, 20_searches_from_scratch) {
  test_search("nim_uniform", 20, {}, nullptr);
}

TEST_F(NimManagerTest, 40_searches_from_4_stones) {
  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake3, nim::kTake2};
  test_search("nim_4_stones", 40, initial_actions, nullptr);
}

TEST_F(NimManagerTest, 40_searches_from_5_stones) {
  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake3, nim::kTake1};
  test_search("nim_5_stones", 40, initial_actions, nullptr);
}

using TicTacToeManagerTest = ManagerTest<TicTacToe>;
TEST_F(TicTacToeManagerTest, uniform_search_log) {

  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  test_search("tictactoe_uniform", 40, initial_actions, nullptr);
}

int main(int argc, char** argv) {
  util::set_tty_mode(false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
