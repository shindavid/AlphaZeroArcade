#include <core/GameServer.hpp>
#include <core/PlayerFactory.hpp>
#include <games/GameTransforms.hpp>
#include <games/nim/Game.hpp>
#include <games/tictactoe/Game.hpp>
#include <games/tictactoe/PlayerFactory.hpp>
#include <games/stochastic_nim/Game.hpp>
#include <games/stochastic_nim/PlayerFactory.hpp>
#include <mcts/SearchLog.hpp>
#include <util/GTestUtil.hpp>

#include <gtest/gtest.h>

template <core::concepts::Game Game, typename PlayerFactory>
class GameServerTest : public testing::Test {
 protected:
  using GameServer = core::GameServer<Game>;
  using GameServerParams = GameServer::Params;
  using TraingDataWriterParams = core::TrainingDataWriter<Game>::Params;
  using action_vec_t = std::vector<core::action_t>;
  using SearchResponse = mcts::Manager<Game>::SearchResponse;
  using SearchResults = Game::Types::SearchResults;

 public:
  GameServerTest() {};

  static GameServerParams create_server_params() {
    return GameServerParams();
  }

  void SetUp() override { util::Random::set_seed(0); }

  void TearDown() override {
    delete server_;
    delete search_log_;
  }

  void init_search(const action_vec_t& initial_actions, int num_iters) {
    GameServerParams server_params = create_server_params();
    server_params.num_game_threads = 1;  // single-threaded for unit tests
    server_params.num_games = 1;         // run only one game
    TraingDataWriterParams training_data_writer_params;
    server_ = new GameServer(server_params, training_data_writer_params, initial_actions);

    std::string player_str1 = std::format(
      "--type=MCTS-C --name=MCTS --no-model --num-search-thread=1 --num-full-iters {}", num_iters);
    std::string player_str2 = "--name=MCTS2 --copy-from=MCTS";
    std::vector<std::string> player_strs = {player_str1, player_str2};

    PlayerFactory player_factory;
    player_factory.set_server(server_);
    auto generator_seats = player_factory.parse(player_strs);
    for (const auto& gen_seat : generator_seats) {
      server_->register_player(gen_seat.seat, gen_seat.generator);
    }

    server_->set_post_setup_hook([this]() {
      auto player = server_->shared_data().get_game_slot(0)->active_player();
      auto mcts_player = dynamic_cast<generic::MctsPlayer<Game>*>(player);
      auto manager = mcts_player->get_manager();

      search_log_ = new mcts::SearchLog<Game>(manager->lookup_table());
      manager->set_post_visit_func([this] { search_log_->update(); });

      mcts_player->set_search_response_processor([this](SearchResponse r) {
        if (is_recorded_) {
          return;
        } else {
          boost_util::pretty_print(ss_result_, r.results->to_json());
          is_recorded_ = true;
        }
      });
    });
  }

  void test_search(const std::string& testname, int num_iters, const action_vec_t& initial_actions) {
    init_search(initial_actions, num_iters);
    server_->run();

    boost::filesystem::path base_dir = util::Repo::root() / "goldenfiles" / "mcts_tests";
    boost::filesystem::path file_path_graph = base_dir / (testname + "_graph.json");
    boost::filesystem::path file_path_result = base_dir / (testname + "_result.json");

    std::ifstream graph_file(file_path_graph);
    std::ifstream result_file(file_path_result);

    std::string expected_graph_json((std::istreambuf_iterator<char>(graph_file)),
                                    std::istreambuf_iterator<char>());
    std::string expected_result_json((std::istreambuf_iterator<char>(result_file)),
                                     std::istreambuf_iterator<char>());

    std::stringstream ss_logs;

    boost_util::pretty_print(ss_logs, search_log_->graphs()[num_iters - 1].graph_repr());

    EXPECT_EQ(ss_logs.str(), expected_graph_json);
    EXPECT_EQ(ss_result_.str(), expected_result_json);

    if (IS_MACRO_ENABLED(WRITE_LOGFILES)) {
      boost::filesystem::path log_dir = util::Repo::root() / "sample_search_logs" / "mcts_tests";
      boost::filesystem::path log_file_path = log_dir / (testname + "_log_debug.json");
      boost_util::write_str_to_file(search_log_->json_str(), log_file_path);
    }
  }

 private:
  GameServer* server_;
  mcts::SearchLog<Game>* search_log_ = nullptr;
  std::stringstream ss_result_;
  bool is_recorded_ = false;
};

using TicTacToeTest = GameServerTest<tictactoe::Game, tictactoe::PlayerFactory>;
using StochasticNimTest = GameServerTest<stochastic_nim::Game, stochastic_nim::PlayerFactory>;

// TEST_F(StochasticNimTest, uniform_search) {
//   std::vector<core::action_t> initial_actions = {
//     stochastic_nim::kTake3, 2, stochastic_nim::kTake3, 2, stochastic_nim::kTake3, 1};

//   test_search("stochastic_nim_uniform_10", 10, initial_actions);
// }

// TEST_F(StochasticNimTest, 20_searches_from_scratch) {
//   test_search("stochastic_nim_uniform", 20, {});
// }

// TEST_F(StochasticNimTest, 100_searches_from_4_stones) {
//   std::vector<core::action_t> initial_actions = {
//     stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0,
//     stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake2, 0};

//   test_search("stochastic_nim_4_stones", 100, initial_actions);
// }

// TEST_F(StochasticNimTest, 100_searches_from_5_stones) {
//   std::vector<core::action_t> initial_actions = {
//     stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0,
//     stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake1, 0};

//   test_search("stochastic_nim_5_stones", 100, initial_actions);
// }

// TEST_F(StochasticNimTest, 100_searches_from_6_stones) {
//   std::vector<core::action_t> initial_actions = {
//     stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0,
//     stochastic_nim::kTake3, 0, stochastic_nim::kTake3, 0};

//   test_search("stochastic_nim_6_stones", 100, initial_actions);
// }

TEST_F(TicTacToeTest, uniform_search) {
  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  test_search("tictactoe_uniform", 40, initial_actions);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
