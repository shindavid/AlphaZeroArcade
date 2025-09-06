#include "alphazero/SearchLog.hpp"
#include "alphazero/SearchResults.hpp"
#include "core/GameServer.hpp"
#include "core/PerfStats.hpp"
#include "games/GameTransforms.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/tictactoe/Game.hpp"
#include "generic_players/MctsPlayer.hpp"
#include "generic_players/MctsPlayerGenerator.hpp"
#include "util/CppUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/RepoUtil.hpp"
#include "util/StringUtil.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

template <core::concepts::Game Game>
class GameServerTest : public testing::Test {
 protected:
  using Traits = a0::Traits<Game>;
  using GameServer = core::GameServer<Game>;
  using GameServerParams = GameServer::Params;
  using action_vec_t = GameServer::action_vec_t;
  using Manager = search::Manager<Traits>;
  using SearchResponse = Manager::SearchResponse;
  using SearchResults = a0::SearchResults<Game>;
  using SearchLog = a0::SearchLog<Traits>;

  // TestPlayer is a simple extension of MctsPlayer. The key differences are:
  //
  // - Records the first MCTS response to a stringstream.
  // - Configures the MCTS manager to record a search log.
  //
  // NOTE[dshin]: I was *tempted* to also incorporate the "initial moves" functionality into
  // TestPlayer. The problem is that for stochastic games, initial moves includes chance-events,
  // which are not executed by AbstractPlayer objects. So it wasn't possible to maintain the
  // existing testing behavior simply by extending MctsPlayer.
  //
  // Also, if we later want to extend this test to operate on multiple concurrent games to test
  // GameServer's multi-threading capabilities, we'll need to better organize the gluing together
  // of the SearchLog/SearchResults. Certainly doable, but no need to do that now.
  class TestPlayer : public generic::MctsPlayer<Game> {
   public:
    using base_t = generic::MctsPlayer<Game>;
    using ActionMask = base_t::ActionMask;
    using ActionResponse = base_t::ActionResponse;

    using base_t::base_t;

    void set_test(GameServerTest* test) {
      test_ = test;

      if (!test->search_log_) {
        auto manager = this->get_manager();
        test->search_log_ = new SearchLog(manager->lookup_table());
        manager->set_post_visit_func([&] { test_->search_log_->update(); });
      }
    }

   protected:
    ActionResponse get_action_response_helper(const SearchResults* results,
                                              const ActionMask& valid_actions) const override {
      if (!test_->is_recorded_) {
        boost_util::pretty_print(test_->ss_result_, results->to_json());
        test_->is_recorded_ = true;
      }
      return base_t::get_action_response_helper(results, valid_actions);
    }

    GameServerTest* test_ = nullptr;
  };

  class TestPlayerGenerator : public generic::MctsPlayerGeneratorBase<Game, TestPlayer> {
   public:
    using base_t = generic::MctsPlayerGeneratorBase<Game, TestPlayer>;

    using base_t::base_t;

    void set_test(GameServerTest* test) { test_ = test; }

    core::AbstractPlayer<Game>* generate(core::game_slot_index_t game_slot_index) override {
      auto player = base_t::generate(game_slot_index);
      dynamic_cast<TestPlayer*>(player)->set_test(test_);
      return player;
    }

   private:
    GameServerTest* test_ = nullptr;
  };

  class TestPlayerSubfactory : public generic::MctsSubfactory<TestPlayerGenerator> {
   public:
    using base_t = generic::MctsSubfactory<TestPlayerGenerator>;

    TestPlayerSubfactory(GameServerTest* test) : test_(test) {}

    TestPlayerGenerator* create(core::GameServerBase* server) override {
      TestPlayerGenerator* generator = base_t::create(server);
      generator->set_test(test_);
      return generator;
    }

   private:
    GameServerTest* test_;
  };

 public:
  GameServerTest() {};

  void SetUp() override {
    util::Random::set_seed(0);
    core::PerfStatsRegistry::clear();
  }

  void TearDown() override {
    delete search_log_;
    delete subfactory_;
    delete server_;
  }

  void init_search(const action_vec_t& initial_actions, int num_iters, int num_threads,
                   const char* model = nullptr) {
    GameServerParams server_params;
    server_params.num_game_threads = 1;
    server_params.num_games = 1;
    server_ = new GameServer(server_params);
    server_->set_initial_actions(initial_actions);

    std::vector<std::string> player_strs = util::split(
      std::format("--num-search-threads={} --num-full-iters {}", num_threads, num_iters));
    if (model) {
      player_strs.push_back(std::format("--model-filename={}", model));
    } else {
      player_strs.push_back("--no-model");
    }

    subfactory_ = new TestPlayerSubfactory(this);
    TestPlayerGenerator* generator1 = subfactory_->create(server_);
    TestPlayerGenerator* generator2 = subfactory_->create(server_);
    generator1->parse_args(player_strs);
    generator2->parse_args(player_strs);

    server_->register_player(-1, generator1);
    server_->register_player(-1, generator2);
  }

  void test_search(const std::string& testname, int num_iters, int num_threads,
                   const action_vec_t& initial_actions, const char* model = nullptr) {
    init_search(initial_actions, num_iters, num_threads, model);
    server_->run();

    boost::filesystem::path base_dir = util::Repo::root() / "goldenfiles" / "gameserver";
    boost::filesystem::path file_path_graph = base_dir / (testname + "_graph.json");
    boost::filesystem::path file_path_result = base_dir / (testname + "_result.json");

    std::stringstream last_snapshot;
    boost_util::pretty_print(last_snapshot, search_log_->graphs()[num_iters - 1].graph_repr());

    if (IS_DEFINED(WRITE_GOLDENFILES)) {
      boost_util::write_str_to_file(ss_result_.str(), file_path_result);
      boost_util::write_str_to_file(last_snapshot.str(), file_path_graph);
    }

    if (IS_DEFINED(WRITE_LOGFILES)) {
      boost::filesystem::path log_dir = util::Repo::root() / "sample_search_logs" / "gameserver";
      boost::filesystem::path log_file_path = log_dir / (testname + "_log.json");
      boost_util::write_str_to_file(search_log_->json_str(), log_file_path);
    }

    std::ifstream graph_file(file_path_graph);
    std::ifstream result_file(file_path_result);

    std::string expected_graph_json((std::istreambuf_iterator<char>(graph_file)),
                                    std::istreambuf_iterator<char>());
    std::string expected_result_json((std::istreambuf_iterator<char>(result_file)),
                                     std::istreambuf_iterator<char>());

    EXPECT_EQ(last_snapshot.str(), expected_graph_json);
    EXPECT_EQ(ss_result_.str(), expected_result_json);
  }

 private:
  friend class TestPlayer;
  TestPlayerSubfactory* subfactory_;
  GameServer* server_;
  SearchLog* search_log_ = nullptr;
  std::stringstream ss_result_;
  bool is_recorded_ = false;
};

using Stochastic_nim = game_transform::AddStateStorage<stochastic_nim::Game>;
using TicTacToe = game_transform::AddStateStorage<tictactoe::Game>;

using TicTacToeTest = GameServerTest<TicTacToe>;
using StochasticNimTest = GameServerTest<Stochastic_nim>;

TEST_F(StochasticNimTest, uniform_search) {
  std::vector<core::action_t> initial_actions = {
    stochastic_nim::kTake3, 2, stochastic_nim::kTake3, 2, stochastic_nim::kTake3, 1};

  test_search("stochastic_nim_uniform_10", 10, 1, initial_actions);
}

TEST_F(StochasticNimTest, 20_searches_from_scratch) {
  test_search("stochastic_nim_uniform", 20, 1, {});
}

TEST_F(StochasticNimTest, 100_searches_from_4_stones) {
  std::vector<core::action_t> initial_actions = {
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0,
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake2, 0};

  test_search("stochastic_nim_4_stones", 100, 1, initial_actions);
}

TEST_F(StochasticNimTest, 100_searches_from_5_stones) {
  std::vector<core::action_t> initial_actions = {
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0,
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake1, 0};

  test_search("stochastic_nim_5_stones", 100, 1, initial_actions);
}

TEST_F(StochasticNimTest, 100_searches_from_6_stones) {
  std::vector<core::action_t> initial_actions = {
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0,
    stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0};

  test_search("stochastic_nim_6_stones", 100, 1, initial_actions);
}

TEST_F(TicTacToeTest, uniform_search) {
  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  test_search("tictactoe_uniform", 40, 1, initial_actions);
}

TEST_F(TicTacToeTest, multi_threaded_uniform_search) {
  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  test_search("tictactoe_multithreaded_uniform", 40, 4, initial_actions,
              "test_models/tictactoe_mini.onnx");
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
