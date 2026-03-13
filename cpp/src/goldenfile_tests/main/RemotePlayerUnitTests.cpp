#include "core/GameServer.hpp"
#include "core/GameServerProxy.hpp"
#include "core/PerfStats.hpp"
#include "games/connect4/Game.hpp"
#include "games/tictactoe/Game.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "util/BoostUtil.hpp"
#include "util/CppUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/RepoUtil.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <thread>

constexpr int kBaseTestPort = 18321;

template <core::concepts::Game Game>
class RemotePlayerTest : public testing::Test {
 protected:
  using GameServer = core::GameServer<Game>;
  using GameServerProxy = core::GameServerProxy<Game>;
  using GameServerParams = typename GameServer::Params;
  using GameServerProxyParams = typename GameServerProxy::Params;
  using results_array_t = typename GameServer::results_array_t;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  void SetUp() override {
    util::Random::set_seed(42);
    core::PerfStatsRegistry::clear();
  }

  results_array_t run_remote_random_vs_random(int num_games, int port) {
    results_array_t server_results;
    std::exception_ptr server_exception;
    std::exception_ptr proxy_exception;

    std::thread server_thread([&]() {
      try {
        GameServerParams server_params;
        server_params.num_games = num_games;
        server_params.parallelism = 1;
        server_params.num_game_threads = 1;
        server_params.port = port;

        GameServer server(server_params);
        server.run();
        server_results = server.get_results();
      } catch (...) {
        server_exception = std::current_exception();
      }
    });

    std::thread proxy_thread([&]() {
      try {
        GameServerProxyParams proxy_params;
        proxy_params.remote_server = "localhost";
        proxy_params.remote_port = port;

        GameServerProxy proxy(proxy_params, 1);

        auto* gen1 = new generic::RandomPlayerGenerator<Game>(&proxy);
        auto* gen2 = new generic::RandomPlayerGenerator<Game>(&proxy);
        proxy.register_player(-1, gen1);
        proxy.register_player(-1, gen2);

        proxy.run();
      } catch (...) {
        proxy_exception = std::current_exception();
      }
    });

    server_thread.join();
    proxy_thread.join();

    if (server_exception) std::rethrow_exception(server_exception);
    if (proxy_exception) std::rethrow_exception(proxy_exception);

    return server_results;
  }

  void test_remote_random_vs_random(const std::string& testname, int num_games, int port) {
    results_array_t results = run_remote_random_vs_random(num_games, port);

    std::stringstream ss;
    for (int p = 0; p < kNumPlayers; ++p) {
      ss << std::format("pid={} {}\n", p, GameServer::get_results_str(results[p]));
    }

    boost::filesystem::path base_dir = util::Repo::root() / "goldenfiles" / "remote_player";
    boost::filesystem::path file_path = base_dir / (testname + "_results.txt");

    if (gtest_util::write_goldenfiles) {
      boost_util::write_str_to_file(ss.str(), file_path);
    }

    std::ifstream file(file_path);
    std::string expected((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    EXPECT_EQ(ss.str(), expected);
  }
};

using TicTacToeRemoteTest = RemotePlayerTest<tictactoe::Game>;
using Connect4RemoteTest = RemotePlayerTest<c4::Game>;

TEST_F(TicTacToeRemoteTest, random_vs_random) {
  test_remote_random_vs_random("tictactoe_random_vs_random", 10, kBaseTestPort);
}

// TEST_F(Connect4RemoteTest, random_vs_random) {
//   test_remote_random_vs_random("c4_random_vs_random", 10, kBaseTestPort + 1);
// }

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
