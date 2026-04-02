#include "core/ActionRequest.hpp"
#include "core/GameServer.hpp"
#include "core/GameServerProxy.hpp"
#include "core/PerfStats.hpp"
#include "games/blokus/Game.hpp"
#include "games/chess/Game.hpp"
#include "games/connect4/Game.hpp"
#include "games/hex/Game.hpp"
#include "games/nim/Game.hpp"
#include "games/othello/Game.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/tictactoe/Game.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "util/BoostUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/Random.hpp"
#include "util/RepoUtil.hpp"
#include "util/mit/mit.hpp"

#include <gtest/gtest.h>

#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#ifdef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be disabled for this test");
#endif

constexpr int kBaseTestPort = 18321;

template <core::concepts::Game Game>
struct ActionLogEntry {
  using Move = Game::Move;

  core::game_id_t game_id;
  core::seat_index_t seat;
  Move move;
};

template <core::concepts::Game Game>
struct ActionLog {
  using ActionLogEntry = ::ActionLogEntry<Game>;

  mit::mutex mutex;
  std::vector<ActionLogEntry> entries;

  void append(ActionLogEntry entry) {
    mit::lock_guard lock(mutex);
    entries.push_back(entry);
  }
};

template <core::concepts::Game Game>
class LoggingRandomPlayer : public generic::RandomPlayer<Game> {
 public:
  using base_t = generic::RandomPlayer<Game>;
  using ActionLog = ::ActionLog<Game>;
  using ActionRequest = core::ActionRequest<Game>;
  using ActionResponse = core::ActionResponse<Game>;

  LoggingRandomPlayer(int base_seed, ActionLog* action_log)
      : base_t(base_seed), action_log_(action_log) {}

  ActionResponse get_action_response(const ActionRequest& request) override {
    ActionResponse response = base_t::get_action_response(request);
    action_log_->append({this->get_game_id(), this->get_my_seat(), response.get_move()});
    return response;
  }

 private:
  ActionLog* action_log_;
};

template <core::concepts::Game Game>
class LoggingRandomPlayerGenerator : public generic::RandomPlayerGenerator<Game> {
 public:
  using base_t = generic::RandomPlayerGenerator<Game>;
  using ActionLog = ::ActionLog<Game>;

  LoggingRandomPlayerGenerator(core::GameServerBase* server, ActionLog* action_log)
      : base_t(server), action_log_(action_log) {}

  core::AbstractPlayer<Game>* generate(core::game_slot_index_t slot_index) override {
    return new LoggingRandomPlayer<Game>(this->base_seed(), action_log_);
  }

 private:
  ActionLog* action_log_;
};

template <core::concepts::Game Game>
class RemotePlayerTest : public testing::Test {
 protected:
  using GameServer = core::GameServer<Game>;
  using GameServerProxy = core::GameServerProxy<Game>;
  using GameServerParams = GameServer::Params;
  using GameServerProxyParams = GameServerProxy::Params;
  using results_array_t = GameServer::results_array_t;
  using ActionLog = ::ActionLog<Game>;
  using ActionLogEntry = ::ActionLogEntry<Game>;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  void SetUp() override {
    util::Random::set_seed(0);
    core::PerfStatsRegistry::clear();
  }

  struct RunResult {
    results_array_t results;
    std::vector<ActionLogEntry> action_log_entries;
  };

  RunResult run_remote_random_vs_random(int num_games, int parallelism, int port) {
    RunResult run_result;
    ActionLog action_log;
    std::exception_ptr server_exception;
    std::exception_ptr proxy_exception;

    int num_remote_players = 1;
    int num_local_players = kNumPlayers - num_remote_players;

    int p = 0;
    mit::thread server_thread([&]() {
      try {
        GameServerParams server_params;
        server_params.num_games = num_games;
        server_params.parallelism = parallelism;
        server_params.num_game_threads = parallelism;
        server_params.port = port;
        server_params.deterministic_mode = true;

        GameServer server(server_params);

        for (; p < num_local_players; ++p) {
          auto* gen = new LoggingRandomPlayerGenerator<Game>(&server, &action_log);
          gen->set_base_seed((p + 1) * 100);
          server.register_player(-1, gen);
        }
        server.run();
        run_result.results = server.get_results();
      } catch (...) {
        server_exception = std::current_exception();
      }
    });

    mit::thread proxy_thread([&]() {
      try {
        GameServerProxyParams proxy_params;
        proxy_params.remote_server = "localhost";
        proxy_params.remote_port = port;

        GameServerProxy proxy(proxy_params, parallelism);

        for (; p < kNumPlayers; ++p) {
          auto* gen = new LoggingRandomPlayerGenerator<Game>(&proxy, &action_log);
          gen->set_base_seed((p + 1) * 100);
          proxy.register_player(-1, gen);
        }

        proxy.run();
      } catch (...) {
        proxy_exception = std::current_exception();
      }
    });

    server_thread.join();
    proxy_thread.join();

    if (server_exception) std::rethrow_exception(server_exception);
    if (proxy_exception) std::rethrow_exception(proxy_exception);

    run_result.action_log_entries = std::move(action_log.entries);
    return run_result;
  }

  static std::string format_action_log(const std::vector<ActionLogEntry>& entries) {
    // Group entries by game_id (deterministic), preserving insertion order within each game.
    std::map<core::game_id_t, std::vector<const ActionLogEntry*>> by_game;
    for (const auto& e : entries) {
      by_game[e.game_id].push_back(&e);
    }

    std::stringstream ss;
    for (const auto& [game_id, game_entries] : by_game) {
      ss << std::format("game={}\n", game_id);
      for (const auto* e : game_entries) {
        ss << std::format("  {} plays {}\n", Game::IO::player_to_str(e->seat),
                          Game::IO::action_to_str(e->action, e->action_mode));
      }
    }
    return ss.str();
  }

  void test_remote_random_vs_random(const std::string& testname, int num_games, int parallelism,
                                    int port) {
    RunResult run_result = run_remote_random_vs_random(num_games, parallelism, port);

    std::stringstream ss;
    for (int p = 0; p < kNumPlayers; ++p) {
      ss << std::format("pid={} {}\n", p, GameServer::get_results_str(run_result.results[p]));
    }
    ss << format_action_log(run_result.action_log_entries);

    boost::filesystem::path base_dir = util::Repo::root() / "goldenfiles" / "remote_player";
    boost::filesystem::path file_path = base_dir / (testname + ".txt");

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
using ChessRemoteTest = RemotePlayerTest<a0achess::Game>;
using HexRemoteTest = RemotePlayerTest<hex::Game>;
using NimRemoteTest = RemotePlayerTest<nim::Game>;
using OthelloRemoteTest = RemotePlayerTest<othello::Game>;
using StochasticNimRemoteTest = RemotePlayerTest<stochastic_nim::Game>;
using BlokusRemoteTest = RemotePlayerTest<blokus::Game>;

TEST_F(TicTacToeRemoteTest, random_vs_random) {
  test_remote_random_vs_random("tictactoe_random_vs_random", 20, 4, kBaseTestPort);
}

TEST_F(Connect4RemoteTest, random_vs_random) {
  test_remote_random_vs_random("c4_random_vs_random", 20, 4, kBaseTestPort + 1);
}

TEST_F(ChessRemoteTest, random_vs_random) {
  test_remote_random_vs_random("chess_random_vs_random", 20, 4, kBaseTestPort + 2);
}

TEST_F(HexRemoteTest, random_vs_random) {
  test_remote_random_vs_random("hex_random_vs_random", 20, 4, kBaseTestPort + 3);
}

TEST_F(NimRemoteTest, random_vs_random) {
  test_remote_random_vs_random("nim_random_vs_random", 20, 4, kBaseTestPort + 4);
}

TEST_F(OthelloRemoteTest, random_vs_random) {
  test_remote_random_vs_random("othello_random_vs_random", 20, 4, kBaseTestPort + 5);
}

TEST_F(StochasticNimRemoteTest, random_vs_random) {
  test_remote_random_vs_random("stochastic_nim_random_vs_random", 20, 4, kBaseTestPort + 6);
}

TEST_F(BlokusRemoteTest, random_vs_random) {
  test_remote_random_vs_random("blokus_random_vs_random", 20, 4, kBaseTestPort + 7);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
