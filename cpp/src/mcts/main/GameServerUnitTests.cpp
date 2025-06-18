#include <core/GameServer.hpp>
#include <core/TrainingDataWriter.hpp>
#include <core/PlayerFactory.hpp>
#include <games/othello/Game.hpp>
#include <games/othello/PlayerFactory.hpp>
#include <games/tictactoe/Game.hpp>
#include <games/tictactoe/PlayerFactory.hpp>
#include <gtest/gtest.h>
#include <util/GTestUtil.hpp>

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

  void SetUp() override {
    util::Random::set_seed(0);
  }

  void TearDown() override { delete server_; }

  GameServer* server() { return server_; }

  void test_search(const action_vec_t& initial_actions) {
    GameServerParams server_params = create_server_params();
    server_params.num_game_threads = 1;  // single-threaded for unit tests
    server_params.num_games = 1;  // run only one game
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
    server_->run();
  }

 private:
  GameServer* server_;
};

using TicTacToeTest = GameServerTest<tictactoe::Game, tictactoe::PlayerFactory>;

TEST_F(TicTacToeTest, uniform_search) {
  std::vector<core::action_t> initial_actions = {0, 1, 2, 4, 7};
  test_search(initial_actions);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
