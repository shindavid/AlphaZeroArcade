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

MctsPlayer* create_mcts_player(Mcts* mcts=nullptr) {
  auto player = new MctsPlayer(MctsPlayer::competitive_params, mcts);
  return player;
}

c4::PerfectPlayer* create_perfect_player() {
  return new c4::PerfectPlayer();
}

player_array_t create_players() {
  return player_array_t{create_mcts_player(), create_perfect_player()};
}

int main(int ac, char* av[]) {
  std::string default_c4_solver_dir_str = util::Config::instance()->get("c4.solver_dir", "");

  namespace po = boost::program_options;
  po::options_description desc("Pit Mcts as red against perfect as yellow");
  desc.add_options()("help,h", "help");

  Mcts::add_options(desc);
  MctsPlayer::competitive_params.add_options(desc, true);
  c4::PerfectPlayParams::add_options(desc, true);

  ParallelGameRunner::global_params.randomize_player_order = false;
  ParallelGameRunner::add_options(desc, true);

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  ParallelGameRunner::global_params.display_progress_bar = true;

  ParallelGameRunner runner;
  runner.register_players([&]() { return create_players(); });
  runner.run();

  MctsPlayer::competitive_params.dump();
  PARAM_DUMP("MCTS search threads", "%d", Mcts::global_params.num_search_threads);
  PARAM_DUMP("MCTS max batch size", "%d", Mcts::global_params.batch_size_limit);
  PARAM_DUMP("MCTS avg batch size", "%.2f", Mcts::global_avg_batch_size());

  return 0;
}
