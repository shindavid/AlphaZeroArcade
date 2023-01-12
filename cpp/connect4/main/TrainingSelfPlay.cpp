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

#include <common/DataExportingMctsPlayer.hpp>
#include <common/GameRunner.hpp>
#include <common/MctsPlayer.hpp>
#include <common/ParallelGameRunner.hpp>
#include <common/TrainingDataWriter.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/StringUtil.hpp>

struct Args {
  std::string games_dir_str;
  int num_mcts_iters_fast;
  int num_mcts_iters_full;
  float full_pct;
};

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;

using TrainingDataWriter = common::TrainingDataWriter<GameState, Tensorizor>;
using ParallelGameRunner = common::ParallelGameRunner<GameState>;
using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
using DataExportingMctsPlayer = common::DataExportingMctsPlayer<GameState, Tensorizor>;
using Mcts = common::Mcts<GameState, Tensorizor>;
using Player = common::AbstractPlayer<GameState>;
using player_array_t = Player::player_array_t;

DataExportingMctsPlayer* create_player(const Args& args, TrainingDataWriter* writer, Mcts* mcts=nullptr) {
  MctsPlayer::Params params;
  params.num_mcts_iters_fast = args.num_mcts_iters_fast;
  params.num_mcts_iters_full = args.num_mcts_iters_full;
  params.full_pct = args.full_pct;
  params.temperature = 0.5;

  auto player = new DataExportingMctsPlayer(writer, params, mcts);
  player->set_name(util::create_string("MCTS-%d", mcts ? 2 : 1));
  return player;
}

player_array_t create_players(const Args& args, TrainingDataWriter* writer) {
  DataExportingMctsPlayer* p1 = create_player(args, writer);
  DataExportingMctsPlayer* p2 = create_player(args, writer, p1->get_mcts());
  return player_array_t{p1, p2};;
}

int main(int ac, char* av[]) {
  Args args;

  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;
  po::options_description desc("Mcts vs mcts");
  desc.add_options()("help,h", "help");

  Mcts::add_options(desc);
  ParallelGameRunner::add_options(desc, true);

  desc.add_options()
      ("games-dir,G", po::value<std::string>(&args.games_dir_str)->default_value("c4_games"),
       "where to write games")
      ("num-mcts-iters-fast,i", po::value<int>(&args.num_mcts_iters_fast)->default_value(100),
       "num mcts iterations to do per fast move")
      ("num-mcts-iters-full,I", po::value<int>(&args.num_mcts_iters_full)->default_value(600),
       "num mcts iterations to do per full move")
      ("full-pct,f", po2::float_value("%.2f", &args.full_pct)->default_value(0.25),
       "pct of moves that should be full ")
      ;

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  ParallelGameRunner runner;
  TrainingDataWriter writer(args.games_dir_str);
  runner.register_players([&]() { return create_players(args, &writer); });
  runner.run();

  printf("MCTS fast-iters:     %6d\n", args.num_mcts_iters_fast);
  printf("MCTS full-iters:     %6d\n", args.num_mcts_iters_full);
  printf("MCTS full-pct:       %6.2f\n", args.full_pct);
  printf("MCTS search threads: %6d\n", Mcts::global_params_.num_search_threads);
  printf("MCTS max batch size: %6d\n", Mcts::global_params_.batch_size_limit);
  printf("MCTS avg batch size: %6.2f\n", Mcts::global_avg_batch_size());

  return 0;
}
