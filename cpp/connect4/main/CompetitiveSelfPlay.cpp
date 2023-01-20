/*
 * TODO: replace all hard-coded c4 classes/functions here with macro values. Then, have the build.py process accept
 * a python file that specifies the game name, along with other relevant metadata (like filesystem paths to the relevant
 * c++ code). The build.py process can then import that file, and pass the macro values to cmake, ultimately causing
 * this file to get compiled to a game-specific binary.
 */
#include <array>
#include <csignal>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <common/GameRunner.hpp>
#include <common/MctsPlayer.hpp>
#include <common/ParallelGameRunner.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/StringUtil.hpp>

namespace po = boost::program_options;

struct Args {
  std::string nnet_filename2;

  auto make_options_description() {
    po::options_description desc("CompetitiveSelfPlay options");

    desc.add_options()
        ("mcts-nnet-filename2", po::value<std::string>(&nnet_filename2)->default_value(""),
         "set this to use a different nnet for the second player")
        ;

    return desc;
  }
};

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;

using ParallelGameRunner = common::ParallelGameRunner<GameState>;
using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
using Mcts = common::Mcts<GameState, Tensorizor>;
using Player = common::AbstractPlayer<GameState>;
using player_array_t = Player::player_array_t;

template<typename... Ts>
MctsPlayer* create_player(int index, Ts&&... ts) {
  MctsPlayer* player = new MctsPlayer(std::forward<Ts>(ts)...);
  player->set_name(util::create_string("MCTS-%d", index));
  return player;
}

player_array_t create_players(
    const Args& args, const MctsPlayer::Params& mcts_player_params, const Mcts::Params& mcts_params)
{
  MctsPlayer* p1 = create_player(1, mcts_player_params, mcts_params);
  MctsPlayer* p2;
  if (!args.nnet_filename2.empty()) {
    Mcts::Params mcts_params2(mcts_params);
    mcts_params2.nnet_filename = args.nnet_filename2;
    p2 = create_player(2, mcts_player_params, mcts_params2);
  } else {
    p2 = create_player(2, mcts_player_params, p1->get_mcts());
  }
  return player_array_t{p1, p2};;
}

ParallelGameRunner::Params get_default_parallel_game_runner_params() {
  ParallelGameRunner::Params parallel_game_runner_params;
  parallel_game_runner_params.randomize_player_order = false;
  parallel_game_runner_params.display_progress_bar = true;
  return parallel_game_runner_params;
}

int main(int ac, char* av[]) {
  po::options_description desc("Mcts vs mcts");
  desc.add_options()("help,h", "help");

  Mcts::Params mcts_params;
  desc.add(mcts_params.make_options_description());

  MctsPlayer::Params mcts_player_params(MctsPlayer::kCompetitive);
  desc.add(mcts_player_params.make_options_description(true));

  ParallelGameRunner::register_signal(SIGTERM);
  ParallelGameRunner::Params parallel_game_runner_params = get_default_parallel_game_runner_params();
  desc.add(parallel_game_runner_params.make_options_description(true));

  Args args;
  desc.add(args.make_options_description());

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  ParallelGameRunner runner(parallel_game_runner_params);
  runner.register_players([&]() { return create_players(args, mcts_player_params, mcts_params); });
  runner.run();

  mcts_player_params.dump();
  PARAM_DUMP("MCTS search threads", "%d", mcts_params.num_search_threads);
  PARAM_DUMP("MCTS max batch size", "%d", mcts_params.batch_size_limit);
  PARAM_DUMP("MCTS avg batch size", "%.2f", Mcts::global_avg_batch_size());

  return 0;
}
