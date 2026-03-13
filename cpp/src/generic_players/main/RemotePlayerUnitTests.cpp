#include "core/GameServer.hpp"
#include "core/GameServerProxy.hpp"
#include "core/PerfStats.hpp"
#include "games/connect4/Game.hpp"
#include "games/tictactoe/Game.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#include <thread>

constexpr int kBaseTestPort = 18321;

template <core::concepts::Game Game>
class RemotePlayerTest : public testing::Test {
 protected:
  using GameServer = core::GameServer<Game>;
  using GameServerProxy = core::GameServerProxy<Game>;
  using GameServerParams = typename GameServer::Params;
  using GameServerProxyParams = typename GameServerProxy::Params;

  void SetUp() override {
    util::Random::set_seed(0);
    core::PerfStatsRegistry::clear();
  }

  void run_remote_random_vs_random(int num_games, int port) {
    std::exception_ptr server_exception;
    std::exception_ptr proxy_exception;

    // Server thread: run GameServer with no local players (all remote).
    // GameServer::wait_for_remote_player_registrations() will fill missing slots with
    // RemotePlayerProxyGenerator, bind/listen on the port, and block on accept().
    std::thread server_thread([&]() {
      try {
        GameServerParams server_params;
        server_params.num_games = num_games;
        server_params.parallelism = 1;
        server_params.num_game_threads = 1;
        server_params.port = port;

        GameServer server(server_params);
        server.run();
      } catch (...) {
        server_exception = std::current_exception();
      }
    });

    // Proxy thread: run GameServerProxy with 2 random players.
    // The GameServerProxy constructor immediately calls create_client_socket(), which has
    // retry logic (5 retries, exponential backoff) to handle the race with the server's listen().
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
  }
};

using TicTacToeRemoteTest = RemotePlayerTest<tictactoe::Game>;
using Connect4RemoteTest = RemotePlayerTest<c4::Game>;

TEST_F(TicTacToeRemoteTest, random_vs_random) {
  run_remote_random_vs_random(2, kBaseTestPort);
}

// TEST_F(Connect4RemoteTest, random_vs_random) {
//   run_remote_random_vs_random(2, kBaseTestPort + 1);
// }

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
