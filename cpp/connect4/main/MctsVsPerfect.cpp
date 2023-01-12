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

struct Args {
  int num_mcts_iters;
  bool verbose;
};

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;

using ParallelGameRunner = common::ParallelGameRunner<GameState>;
using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
using Mcts = common::Mcts<GameState, Tensorizor>;
using Player = common::AbstractPlayer<GameState>;
using player_array_t = Player::player_array_t;

MctsPlayer* create_mcts_player(const Args& args, Mcts* mcts=nullptr) {
  MctsPlayer::Params params;
  params.num_mcts_iters_full = args.num_mcts_iters;
  params.full_pct = 1.0;
  params.temperature = 0;
  params.verbose = args.verbose;
  auto player = new MctsPlayer(params, mcts);
  player->set_name(util::create_string("MCTS-m%d", args.num_mcts_iters));
  return player;
}

c4::PerfectPlayer* create_perfect_player(const Args& args) {
  return new c4::PerfectPlayer();
}

player_array_t create_players(const Args& args) {
  return player_array_t{create_mcts_player(args), create_perfect_player(args)};
}

int main(int ac, char* av[]) {
  Args args;

  std::string default_c4_solver_dir_str = util::Config::instance()->get("c4.solver_dir", "");

  namespace po = boost::program_options;
  po::options_description desc("Pit Mcts as red against perfect as yellow");
  desc.add_options()("help,h", "help");

  Mcts::global_params_.dirichlet_mult = 0;
  Mcts::add_options(desc);

  c4::PerfectPlayParams::add_options(desc, true);

  ParallelGameRunner::global_params_.randomize_player_order = false;
  ParallelGameRunner::add_options(desc, true);

  desc.add_options()
      ("num-mcts-iters,m", po::value<int>(&args.num_mcts_iters)->default_value(400), "num mcts iterations to do per move")
      ("verbose,v", po::bool_switch(&args.verbose)->default_value(false), "verbose")
      ;

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  ParallelGameRunner runner;
  runner.register_players([&]() { return create_players(args); });
  runner.run();

  printf("MCTS iters:          %6d\n", args.num_mcts_iters);
  printf("MCTS search threads: %6d\n", Mcts::global_params_.num_search_threads);
  printf("MCTS max batch size: %6d\n", Mcts::global_params_.batch_size_limit);
  printf("MCTS avg batch size: %6.2f\n", Mcts::global_avg_batch_size());

  return 0;
}
