#include <array>
#include <iostream>

#include <boost/program_options.hpp>

#include <common/GameServer.hpp>
#include <common/MctsPlayer.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/ParamDumper.hpp>

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;

using GameServer = common::GameServer<GameState>;
using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
using Mcts = common::Mcts<GameState, Tensorizor>;
using Player = common::AbstractPlayer<GameState>;
using player_array_t = Player::player_array_t;

GameServer::Params get_default_game_server_params() {
  GameServer::Params parallel_game_runner_params;
  parallel_game_runner_params.display_progress_bar = true;
  return parallel_game_runner_params;
}

int main(int ac, char* av[]) {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  Mcts::Params mcts_params(Mcts::kCompetitive);
  MctsPlayer::Params mcts_player_params(MctsPlayer::kCompetitive);
  c4::PerfectPlayParams perfect_play_params;
//  ParallelGameRunner::register_signal(SIGTERM);
  GameServer::Params game_server_params = get_default_game_server_params();

  po2::options_description raw_desc("General options");
  auto desc = raw_desc.template add_option<"help", 'h'>("help")
      .add(mcts_params.make_options_description())
      .add(mcts_player_params.make_options_description())
      .add(perfect_play_params.make_options_description())
      .add(game_server_params.make_options_description());

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  GameServer server(game_server_params);

  // mcts player should always be seated at seat 0
  server.register_player(0, [&]() { return new MctsPlayer(mcts_player_params, mcts_params); });
  server.register_player([&]() { return new c4::PerfectPlayer(perfect_play_params); });
  server.run();

  mcts_player_params.dump();
  util::ParamDumper::add("MCTS search threads", "%d", mcts_params.num_search_threads);
  util::ParamDumper::add("MCTS max batch size", "%d", mcts_params.batch_size_limit);
  util::ParamDumper::add("MCTS avg batch size", "%.2f", Mcts::global_avg_batch_size());
  util::ParamDumper::flush();

  return 0;
}
