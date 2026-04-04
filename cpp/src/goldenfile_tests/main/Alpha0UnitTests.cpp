#include "alpha0/ManagerParams.hpp"
#include "alpha0/Traits.hpp"
#include "core/BasicTypes.hpp"
#include "core/Constants.hpp"
#include "core/EvalSpecTransforms.hpp"
#include "core/GameServerBase.hpp"
#include "games/nim/Bindings.hpp"
#include "games/nim/Game.hpp"
#include "games/stochastic_nim/Bindings.hpp"
#include "games/stochastic_nim/Constants.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/tictactoe/Bindings.hpp"
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
  using MoveSet = Game::MoveSet;
  using Base = search::SimpleNNEvaluationService<Traits>;
  using NNEvaluation = search::NNEvaluation<Traits>;
  using GameResultTensor = GameTypes::GameResultTensor;
  using PolicyTensor = GameTypes::PolicyTensor;
  using ActionValueTensor = GameTypes::ActionValueTensor;
  using Item = Base::Item;

  MockNNEvaluationService(bool smart) : smart_(smart) {
    this->set_init_func([&](NNEvaluation* eval, const Item& item) { this->init_eval(eval, item); });
  }

  void init_eval(NNEvaluation* eval, const Item& item) {
    GameResultTensor value;
    PolicyTensor policy;
    ActionValueTensor action_values;
    group::element_t sym = group::kIdentity;

    core::seat_index_t seat = item.node()->stable_data().active_seat;
    core::game_phase_t phase = item.node()->game_phase();

    auto tensorizor = item.input_tensorizor();
    const State& state = tensorizor->current_frame();
    action_values.setZero();

    bool winning = state.stones_left % (1 + nim::kMaxStonesToTake) != 0;
    if (winning) {
      int winning_move = state.stones_left % (1 + nim::kMaxStonesToTake) - 1;

      float winning_action_p = smart_ ? 2 : 0;
      float losing_action_p = smart_ ? 0 : 2;

      // these are logits
      float winning_v = smart_ ? 2 : 0;
      float losing_v = smart_ ? 0 : 2;

      value.setValues({winning_v, losing_v});

      policy.setConstant(losing_action_p);
      policy(winning_move) = winning_action_p;

      action_values(winning_move, 0) = winning_v;
    } else {
      value.setZero();
      policy.setZero();
    }

    auto outputs = std::make_tuple(policy, value, action_values);
    MoveSet valid_moves = Game::Rules::analyze(state).valid_moves();
    eval->init(outputs, valid_moves, sym, seat, phase);
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
  using Move = Game::Move;
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

  void start_manager(const std::vector<Move>& initial_moves = {}) {
    manager_->start();
    for (Move move : initial_moves) {
      manager_->update(move);
    }
    this->initial_moves_ = initial_moves;
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
                   const std::vector<Move>& initial_moves, Service_sptr service) {
    init_manager(service);
    start_manager(initial_moves);
    const SearchResults* result = search(num_search);

    auto root = util::Repo::root();
    boost::filesystem::path base_dir = root / "goldenfiles" / "alpha0_tests";

    boost::filesystem::path file_path_result = base_dir / (testname + "_result.json");
    boost::filesystem::path file_path_graph = base_dir / (testname + "_graph.json");

    std::stringstream ss_result;
    boost_util::pretty_print(ss_result, result->to_json());

    if (gtest_util::write_goldenfiles) {
      boost_util::write_str_to_file(ss_result.str(), file_path_result);
      boost_util::write_str_to_file(get_search_log()->last_graph_json_str(), file_path_graph);

      boost::filesystem::path log_dir = root / "sample_search_logs" / "alpha0_tests";
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
  std::vector<Move> initial_moves_;
  SearchLog* search_log_ = nullptr;
};

using NimManagerTest = ManagerTest<NimTraits>;
TEST_F(NimManagerTest, uniform_search) {
  std::vector<Move> initial_moves = {nim::kTake3, nim::kTake3, nim::kTake3,
                                     nim::kTake3, nim::kTake3, nim::kTake2};
  test_search("nim_uniform_10", 10, initial_moves, nullptr);
}

TEST_F(NimManagerTest, smart_search) {
  std::shared_ptr<MockNNEvaluationService<NimTraits>> mock_service =
    std::make_shared<MockNNEvaluationService<NimTraits>>(true);

  std::vector<Move> initial_moves = {nim::kTake3, nim::kTake3, nim::kTake3,
                                     nim::kTake3, nim::kTake3, nim::kTake2};
  test_search("nim_smart_service", 10, initial_moves, mock_service);
}

TEST_F(NimManagerTest, dumb_search) {
  std::shared_ptr<MockNNEvaluationService<NimTraits>> mock_service =
    std::make_shared<MockNNEvaluationService<NimTraits>>(false);

  std::vector<Move> initial_moves = {nim::kTake3, nim::kTake3, nim::kTake3,
                                     nim::kTake3, nim::kTake3, nim::kTake2};

  test_search("nim_dumb_service", 10, initial_moves, mock_service);
}

TEST_F(NimManagerTest, 20_searches_from_scratch) { test_search("nim_uniform", 20, {}, nullptr); }

TEST_F(NimManagerTest, 40_searches_from_4_stones) {
  std::vector<nim::Move> initial_moves = {nim::kTake3, nim::kTake3, nim::kTake3,
                                          nim::kTake3, nim::kTake3, nim::kTake2};
  test_search("nim_4_stones", 40, initial_moves, nullptr);
}

TEST_F(NimManagerTest, 40_searches_from_5_stones) {
  std::vector<nim::Move> initial_moves = {nim::kTake3, nim::kTake3, nim::kTake3,
                                          nim::kTake3, nim::kTake3, nim::kTake1};
  test_search("nim_5_stones", 40, initial_moves, nullptr);
}

using StochasticNimManagerTest = ManagerTest<StochasticNimTraits>;
TEST_F(StochasticNimManagerTest, uniform_search) {
  std::vector<stochastic_nim::Move> initial_moves = {
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(2, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(2, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(1, stochastic_nim::kChancePhase),
  };

  test_search("stochastic_nim_uniform_10", 10, initial_moves, nullptr);
}

TEST_F(StochasticNimManagerTest, 20_searches_from_scratch) {
  test_search("stochastic_nim_uniform", 20, {}, nullptr);
}

TEST_F(StochasticNimManagerTest, 100_searches_from_4_stones) {
  std::vector<stochastic_nim::Move> initial_moves = {
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake2, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase)};

  test_search("stochastic_nim_4_stones", 100, initial_moves, nullptr);
}

TEST_F(StochasticNimManagerTest, 100_searches_from_5_stones) {
  std::vector<stochastic_nim::Move> initial_moves = {
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake1, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase)};

  test_search("stochastic_nim_5_stones", 100, initial_moves, nullptr);
}

TEST_F(StochasticNimManagerTest, 100_searches_from_6_stones) {
  std::vector<stochastic_nim::Move> initial_moves = {
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase)};

  test_search("stochastic_nim_6_stones", 100, initial_moves, nullptr);
}

using TicTacToeManagerTest = ManagerTest<TicTacToeTraits>;
TEST_F(TicTacToeManagerTest, uniform_search_log) {
  std::vector<tictactoe::Move> initial_moves = {0, 1, 2, 4, 7};
  test_search("tictactoe_uniform", 40, initial_moves, nullptr);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
