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

DataExportingMctsPlayer* create_player(TrainingDataWriter* writer, Mcts* mcts=nullptr) {
  auto player = new DataExportingMctsPlayer(writer, MctsPlayer::training_params, mcts);
  player->set_name(util::create_string("MCTS-%d", mcts ? 2 : 1));
  return player;
}

player_array_t create_players(TrainingDataWriter* writer) {
  DataExportingMctsPlayer* p1 = create_player(writer);
  DataExportingMctsPlayer* p2 = create_player(writer, p1->get_mcts());
  return player_array_t{p1, p2};;
}

int main(int ac, char* av[]) {
  Args args;

  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;
  po::options_description desc("Mcts vs mcts");
  desc.add_options()("help,h", "help");

  Mcts::add_options(desc);
  MctsPlayer::training_params.add_options(desc, true);

  ParallelGameRunner::global_params.num_games = 10000;
  ParallelGameRunner::global_params.randomize_player_order = false;
  ParallelGameRunner::global_params.display_progress_bar = true;

  ParallelGameRunner::add_options(desc, true);

  desc.add_options()
      ("games-dir,g", po::value<std::string>(&args.games_dir_str)->default_value("c4_games"),
       "where to write games")
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
  runner.register_players([&]() { return create_players(&writer); });
  runner.run();

  MctsPlayer::training_params.dump();
  PARAM_DUMP("MCTS search threads", "%d", Mcts::global_params.num_search_threads);
  PARAM_DUMP("MCTS max batch size", "%d", Mcts::global_params.batch_size_limit);
  PARAM_DUMP("MCTS avg batch size", "%.2f", Mcts::global_avg_batch_size());

  return 0;
}
