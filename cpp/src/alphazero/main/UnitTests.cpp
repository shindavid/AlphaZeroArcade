#include "alphazero/ManagerParams.hpp"
#include "alphazero/Traits.hpp"
#include "core/BasicTypes.hpp"
#include "core/Constants.hpp"
#include "core/EvalSpecTransforms.hpp"
#include "core/GameServerBase.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "games/nim/Game.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/tictactoe/Game.hpp"
#include "search/LookupTable.hpp"
#include "search/Manager.hpp"
#include "search/NNEvaluation.hpp"
#include "search/SearchLog.hpp"
#include "search/SearchParams.hpp"
#include "search/SearchRequest.hpp"
#include "search/SimpleNNEvaluationService.hpp"
#include "search/TraitsTypes.hpp"
#include "util/BoostUtil.hpp"
#include "util/CppUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/RepoUtil.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using NimSpec = transforms::AddStateStorage<core::EvalSpec<nim::Game, core::kParadigmAlphaZero>>;
using StochasticNimSpec =
  transforms::AddStateStorage<core::EvalSpec<stochastic_nim::Game, core::kParadigmAlphaZero>>;
using TicTacToeSpec =
  transforms::AddStateStorage<core::EvalSpec<tictactoe::Game, core::kParadigmAlphaZero>>;

using NimTraits = alpha0::Traits<nim::Game, NimSpec>;
using StochasticNimTraits = alpha0::Traits<stochastic_nim::Game, StochasticNimSpec>;
using TicTacToeTraits = alpha0::Traits<tictactoe::Game, TicTacToeSpec>;

template <search::concepts::Traits Traits>
class MockNNEvaluationService : public search::SimpleNNEvaluationService<Traits> {
 public:
  using Game = Traits::Game;
  using GameTypes = Game::Types;
  using State = Game::State;
  using Base = search::SimpleNNEvaluationService<Traits>;
  using NNEvaluation = search::NNEvaluation<Traits>;
  using ValueTensor = GameTypes::ValueTensor;
  using PolicyTensor = GameTypes::PolicyTensor;
  using ActionValueTensor = GameTypes::ActionValueTensor;
  using ActionMask = GameTypes::ActionMask;
  using Item = Base::Item;

  MockNNEvaluationService(bool smart) : smart_(smart) {
    this->set_init_func([&](NNEvaluation* eval, const Item& item) { this->init_eval(eval, item); });
  }

  void init_eval(NNEvaluation* eval, const Item& item) {
    ValueTensor value;
    PolicyTensor policy;
    ActionValueTensor action_values;
    group::element_t sym = group::kIdentity;

    ActionMask valid_actions = item.node()->stable_data().valid_action_mask;
    core::seat_index_t seat = item.node()->stable_data().active_seat;
    core::action_mode_t mode = item.node()->action_mode();

    const State& state = item.cur_state();

    bool winning = state.stones_left % (1 + nim::kMaxStonesToTake) != 0;
    if (winning) {
      core::action_t winning_move = state.stones_left % (1 + nim::kMaxStonesToTake) - 1;

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

    auto outputs = std::make_tuple(policy, value, action_values);
    eval->init(outputs, valid_actions, sym, seat, mode);
  }

 private:
  bool smart_;
};

template <search::concepts::Traits Traits>
class ManagerTest : public testing::Test {
 protected:
  using EvalSpec = Traits::EvalSpec;
  using Game = Traits::Game;
  using TraitsTypes = search::TraitsTypes<Traits>;
  using Manager = search::Manager<Traits>;
  using ManagerParams = alpha0::ManagerParams<EvalSpec>;
  using Node = TraitsTypes::Node;
  using Edge = Traits::Edge;
  using StateHistory = Game::StateHistory;
  using action_t = core::action_t;
  using LookupTable = search::LookupTable<Traits>;
  using ValueArray = Game::Types::ValueArray;
  using Service = search::NNEvaluationServiceBase<Traits>;
  using Service_sptr = Service::sptr;
  using State = Game::State;
  using SearchResults = Traits::SearchResults;
  using SearchLog = search::SearchLog<Traits>;

  static_assert(core::kStoreStates<EvalSpec>, "state-storage required for search-log tests");

 public:
  ManagerTest() : manager_params_(create_manager_params()) {}

  ~ManagerTest() override {
    // delete manager_;
  }

  static ManagerParams create_manager_params() {
    ManagerParams params(search::kCompetition);
    params.no_model = true;
    return params;
  }

  void SetUp() override { util::Random::set_seed(0); }

  void init_manager(Service_sptr service = nullptr) {
    core::GameServerBase* server = nullptr;
    manager_ = new Manager(manager_params_, server, service);
    search_log_ = new SearchLog(manager_->lookup_table());
    manager_->set_post_visit_func([&] { search_log_->update(); });
  }

  void start_manager(const std::vector<core::action_t>& initial_actions = {}) {
    manager_->start();
    for (core::action_t action : initial_actions) {
      manager_->update(action);
    }
    this->initial_actions_ = initial_actions;
  }

  ManagerParams& manager_params() { return manager_params_; }

  const SearchResults* search(int num_searches = 0) {
    search::SearchParams search_params(num_searches, true);
    manager_->set_search_params(search_params);
    search::SearchRequest request;
    return manager_->search(request).results;
  }

  Node* get_node_by_index(core::node_pool_index_t index) {
    return manager_->shared_data()->lookup_table.get_node(index);
  }

  SearchLog* get_search_log() { return search_log_; }
  ManagerParams& get_manager_params() { return manager_params_; }

  void test_search(const std::string& testname, int num_search,
                   const std::vector<core::action_t>& initial_actions, Service_sptr service) {
    init_manager(service);
    start_manager(initial_actions);
    const SearchResults* result = search(num_search);

    auto root = util::Repo::root();
    boost::filesystem::path base_dir = root / "goldenfiles" / "alphazero_tests";

    boost::filesystem::path file_path_result = base_dir / (testname + "_result.json");
    boost::filesystem::path file_path_graph = base_dir / (testname + "_graph.json");

    std::stringstream ss_result;
    boost_util::pretty_print(ss_result, result->to_json());

    if (IS_DEFINED(WRITE_GOLDENFILES)) {
      boost_util::write_str_to_file(ss_result.str(), file_path_result);
      boost_util::write_str_to_file(get_search_log()->last_graph_json_str(), file_path_graph);
    }

    if (IS_DEFINED(WRITE_LOGFILES)) {
      boost::filesystem::path log_dir = root / "sample_search_logs" / "alphazero_tests";
      boost::filesystem::path log_file_path = log_dir / (testname + "_log.json");
      boost_util::write_str_to_file(get_search_log()->json_str(), log_file_path);
    }

    std::ifstream result_file(file_path_result);
    std::ifstream graph_file(file_path_graph);

    std::string expected_result_json((std::istreambuf_iterator<char>(result_file)),
                                     std::istreambuf_iterator<char>());
    std::string expected_graph_json((std::istreambuf_iterator<char>(graph_file)),
                                    std::istreambuf_iterator<char>());

    EXPECT_EQ(ss_result.str(), expected_result_json);
    EXPECT_EQ(get_search_log()->last_graph_json_str(), expected_graph_json);
  }

 private:
  ManagerParams manager_params_;
  Manager* manager_ = nullptr;
  std::vector<core::action_t> initial_actions_;
  SearchLog* search_log_ = nullptr;
};

using NimManagerTest = ManagerTest<NimTraits>;
TEST_F(NimManagerTest, uniform_search) {
  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake3, nim::kTake2};
  test_search("nim_uniform_10", 10, initial_actions, nullptr);
}

TEST_F(NimManagerTest, smart_search) {
  std::shared_ptr<MockNNEvaluationService<NimTraits>> mock_service =
    std::make_shared<MockNNEvaluationService<NimTraits>>(true);

  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake3, nim::kTake2};
  test_search("nim_smart_service", 10, initial_actions, mock_service);
}

TEST_F(NimManagerTest, dumb_search) {
  std::shared_ptr<MockNNEvaluationService<NimTraits>> mock_service =
    std::make_shared<MockNNEvaluationService<NimTraits>>(false);

  std::vector<core::action_t> initial_actions = {nim::kTake3, nim::kTake3, nim::kTake3,
                                                 nim::kTake3, nim::kTake3, nim::kTake2};

  test_search("nim_dumb_service", 10, initial_actions, mock_service);
}

TEST_F(NimManagerTest, 20_searches_from_scratch) { test_search("nim_uniform", 20, {}, nullptr); }

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

using StochasticNimManagerTest = ManagerTest<StochasticNimTraits>;
TEST_F(StochasticNimManagerTest, uniform_search) {
  std::vector<core::action_t> initial_actions = {
    stochastic_nim::kTake3, 2, stochastic_nim::kTake3, 2, stochastic_nim::kTake3, 1};

  test_search("stochastic_nim_uniform_10", 10, initial_actions, nullptr);
}

TEST_F(StochasticNimManagerTest, 20_searches_from_scratch) {
  test_search("stochastic_nim_uniform", 20, {}, nullptr);
}

TEST_F(StochasticNimManagerTest, 100_searches_from_4_stones) {
  std::vector<core::action_t> initial_actions = {
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0,
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake2, 0};

  test_search("stochastic_nim_4_stones", 100, initial_actions, nullptr);
}

TEST_F(StochasticNimManagerTest, 100_searches_from_5_stones) {
  std::vector<core::action_t> initial_actions = {
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0,
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake1, 0};

  test_search("stochastic_nim_5_stones", 100, initial_actions, nullptr);
}

TEST_F(StochasticNimManagerTest, 100_searches_from_6_stones) {
  std::vector<core::action_t> initial_actions = {
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0,
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0};

  test_search("stochastic_nim_6_stones", 100, initial_actions, nullptr);
}

using TicTacToeManagerTest = ManagerTest<TicTacToeTraits>;
TEST_F(TicTacToeManagerTest, uniform_search_log) {
  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  test_search("tictactoe_uniform", 40, initial_actions, nullptr);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
