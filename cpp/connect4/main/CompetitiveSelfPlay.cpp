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
  std::string nnet_filename2;
};

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;

using ParallelGameRunner = common::ParallelGameRunner<GameState>;
using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
using Mcts = common::Mcts<GameState, Tensorizor>;
using Player = common::AbstractPlayer<GameState>;
using player_array_t = Player::player_array_t;

MctsPlayer* create_player(const Args& args, Mcts* mcts=nullptr) {
  MctsPlayer* player;
  if (args.nnet_filename2.empty() || !mcts) {
    player = new MctsPlayer(MctsPlayer::competitive_params, mcts);
  } else {
    Mcts::Params mcts_params = Mcts::global_params;
    mcts_params.nnet_filename = args.nnet_filename2;
    player = new MctsPlayer(MctsPlayer::competitive_params, mcts_params);
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

  Mcts::add_options(desc);
  MctsPlayer::competitive_params.add_options(desc, true);
  ParallelGameRunner::add_options(desc, true);

  desc.add_options()
      ("mcts-nnet-filename2", po::value<std::string>(&args.nnet_filename2)->default_value(""),
       "set this to use a different nnet for the second player")
      ;

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  ParallelGameRunner::global_params.randomize_player_order = false;
  ParallelGameRunner::global_params.display_progress_bar = true;

  ParallelGameRunner runner;
  runner.register_players([&]() { return create_players(args); });
  runner.run();

  MctsPlayer::competitive_params.dump();
  PARAM_DUMP("MCTS search threads", "%d", Mcts::global_params.num_search_threads);
  PARAM_DUMP("MCTS max batch size", "%d", Mcts::global_params.batch_size_limit);
  PARAM_DUMP("MCTS avg batch size", "%.2f", Mcts::global_avg_batch_size());

  return 0;
}
