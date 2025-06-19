#include <core/GameServer.hpp>
#include <core/PlayerFactory.hpp>
#include <games/tictactoe/Game.hpp>
#include <games/tictactoe/PlayerFactory.hpp>
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

  void init_search(const action_vec_t& initial_actions) {
    GameServerParams server_params = create_server_params();
    server_params.num_game_threads = 1;  // single-threaded for unit tests
    server_params.num_games = 1;         // run only one game
    TraingDataWriterParams training_data_writer_params;
    server_ = new GameServer(server_params, training_data_writer_params, initial_actions);

    std::string player_str1 = "--type=MCTS-C --name=MCTS --no-model --num-search-thread=1";
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
    });
  }

  void test_search(const std::string& testname, const action_vec_t& initial_actions) {
    init_search(initial_actions);
    server_->run();

    boost::filesystem::path base_dir = util::Repo::root() / "goldenfiles" / "mcts_tests";

    boost::filesystem::path file_path_graph =
        base_dir / (testname + "_graph.json");

    std::ifstream graph_file(file_path_graph);

    std::string expected_graph_json((std::istreambuf_iterator<char>(graph_file)),
                                  std::istreambuf_iterator<char>());
    std::stringstream ss;
    boost_util::pretty_print(ss, search_log_->graphs()[39].graph_repr());
    EXPECT_EQ(ss.str(), expected_graph_json);
  }

 private:
  GameServer* server_;
  mcts::SearchLog<Game>* search_log_ = nullptr;
};

using TicTacToeTest = GameServerTest<tictactoe::Game, tictactoe::PlayerFactory>;

TEST_F(TicTacToeTest, uniform_search) {
  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  test_search("tictactoe_uniform", initial_actions);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
