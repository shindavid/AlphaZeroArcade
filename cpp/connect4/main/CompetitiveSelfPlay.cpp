/*
 * TODO: replace all hard-coded c4 classes/functions here with macro values. Then, have the build.py process accept
 * a python file that specifies the game name, along with other relevant metadata (like filesystem paths to the relevant
 * c++ code). The build.py process can then import that file, and pass the macro values to cmake, ultimately causing
 * this file to get compiled to a game-specific binary.
 */
#include <array>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <common/GameRunner.hpp>
#include <common/MctsPlayer.hpp>
#include <common/ParallelGameRunner.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/StringUtil.hpp>

struct Args {
  std::string games_dir_str;
  std::string nnet_filename2;
  int num_mcts_iters;
};

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;

using ParallelGameRunner = common::ParallelGameRunner<GameState>;
using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
using Mcts = common::Mcts<GameState, Tensorizor>;
using Player = common::AbstractPlayer<GameState>;
using player_array_t = Player::player_array_t;

MctsPlayer* create_player(const Args& args, Mcts* mcts=nullptr) {
  MctsPlayer::Params params;
  params.num_mcts_iters = args.num_mcts_iters;
  params.temperature = 0.0;

  MctsPlayer* player;
  if (args.nnet_filename2.empty() || !mcts) {
    player = new MctsPlayer(params, mcts);
  } else {
    Mcts::Params mcts_params = Mcts::global_params_;
    mcts_params.nnet_filename = args.nnet_filename2;
    player = new MctsPlayer(params, mcts_params);
  }
  player->set_name(util::create_string("MCTS-%d", mcts ? 2 : 1));
  return player;
}

player_array_t create_players(const Args& args) {
  MctsPlayer* p1 = create_player(args);
  MctsPlayer* p2 = create_player(args, p1->get_mcts());
  return player_array_t{p1, p2};;
}

int main(int ac, char* av[]) {
  Args args;

  namespace po = boost::program_options;
  po::options_description desc("Mcts vs mcts");
  desc.add_options()("help,h", "help");

  Mcts::global_params_.dirichlet_mult = 0;
  Mcts::add_options(desc);
  ParallelGameRunner::add_options(desc, true);

  desc.add_options()
      ("games-dir,G", po::value<std::string>(&args.games_dir_str)->default_value("c4_games"),
       "where to write games (only if --export/-x option is set)")
      ("mcts-nnet-filename2", po::value<std::string>(&args.nnet_filename2)->default_value(""),
       "set this to use a different nnet for the second player")
      ("num-mcts-iters,m", po::value<int>(&args.num_mcts_iters)->default_value(400), "num mcts iterations to do per move")
      ;

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  ParallelGameRunner::global_params_.randomize_player_order = true;

  ParallelGameRunner runner;
  runner.register_players([&]() { return create_players(args); });
  runner.run();

  printf("MCTS iters:          %6d\n", args.num_mcts_iters);
  printf("MCTS search threads: %6d\n", Mcts::global_params_.num_search_threads);
  printf("MCTS max batch size: %6d\n", Mcts::global_params_.batch_size_limit);
  printf("MCTS avg batch size: %6.2f\n", Mcts::global_avg_batch_size());

  return 0;
}
