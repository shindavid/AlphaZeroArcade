#include <array>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <common/GameRunner.hpp>
#include <common/MctsPlayer.hpp>
#include <common/ParallelGameRunner.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/Config.hpp>
#include <util/StringUtil.hpp>

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;

using ParallelGameRunner = common::ParallelGameRunner<GameState>;
using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
using Mcts = common::Mcts<GameState, Tensorizor>;
using Player = common::AbstractPlayer<GameState>;
using player_array_t = Player::player_array_t;

player_array_t create_players(
    const MctsPlayer::Params& mcts_player_params, const Mcts::Params& mcts_params,
    const c4::PerfectPlayParams& perfect_play_params)
{
  return player_array_t{
    new MctsPlayer(mcts_player_params, mcts_params),
    new c4::PerfectPlayer(perfect_play_params)
  };
}

ParallelGameRunner::Params get_default_parallel_game_runner_params() {
  ParallelGameRunner::Params parallel_game_runner_params;
  parallel_game_runner_params.randomize_player_order = false;
  parallel_game_runner_params.display_progress_bar = true;
  return parallel_game_runner_params;
}

int main(int ac, char* av[]) {
  std::string default_c4_solver_dir_str = util::Config::instance()->get("c4.solver_dir", "");

  namespace po = boost::program_options;
  po::options_description desc("Pit Mcts as red against perfect as yellow");
  desc.add_options()("help,h", "help");

  Mcts::Params mcts_params;
  desc.add(mcts_params.make_options_description());

  MctsPlayer::Params mcts_player_params(MctsPlayer::kCompetitive);
  desc.add(mcts_player_params.make_options_description(true));

  c4::PerfectPlayParams perfect_play_params;
  desc.add(perfect_play_params.make_options_description(true));

  ParallelGameRunner::register_signal(SIGTERM);
  ParallelGameRunner::Params parallel_game_runner_params = get_default_parallel_game_runner_params();
  desc.add(parallel_game_runner_params.make_options_description(true));

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  ParallelGameRunner runner(parallel_game_runner_params);
  runner.register_players([&]() { return create_players(mcts_player_params, mcts_params, perfect_play_params); });
  runner.run();

  mcts_player_params.dump();
  PARAM_DUMP("MCTS search threads", "%d", mcts_params.num_search_threads);
  PARAM_DUMP("MCTS max batch size", "%d", mcts_params.batch_size_limit);
  PARAM_DUMP("MCTS avg batch size", "%.2f", Mcts::global_avg_batch_size());

  return 0;
}
