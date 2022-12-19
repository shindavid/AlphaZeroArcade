#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <common/DerivedTypes.hpp>
#include <common/GameRunner.hpp>
#include <common/NNetPlayer.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/Config.hpp>
#include <util/StringUtil.hpp>

struct Args {
  std::string c4_solver_dir_str;
  int num_mcts_iters;
  int num_search_threads;
  int batch_size_limit;
  int num_games;
  std::string mcts_profiling_dir;
};

using C4NNetPlayer = common::NNetPlayer<c4::GameState, c4::Tensorizor>;
using Mcts = common::Mcts_<c4::GameState, c4::Tensorizor>;

C4NNetPlayer* create_nnet_player(const Args& args) {
  C4NNetPlayer::Params params;
  params.num_mcts_iters = args.num_mcts_iters;
  params.num_search_threads = args.num_search_threads;
  params.batch_size_limit = args.batch_size_limit;
  params.temperature = 0;
  auto player = new C4NNetPlayer(params);
  player->set_name(util::create_string("MCTS-m%d-s%d", args.num_mcts_iters, args.num_search_threads));
  return player;
}

c4::PerfectPlayer* create_perfect_player(const Args& args) {
  if (args.c4_solver_dir_str.empty()) {
    throw util::Exception("Must either pass -c or add an entry for \"c4.solver_dir\" to %s",
                          util::Config::instance()->config_path().c_str());
  }

  boost::filesystem::path c4_solver_dir(args.c4_solver_dir_str);
  c4::PerfectPlayer::Params params;
  params.c4_solver_dir = c4_solver_dir;
  return new c4::PerfectPlayer(params);
}

int main(int ac, char* av[]) {
  Args args;

  std::string default_c4_solver_dir_str = util::Config::instance()->get("c4.solver_dir", "");

  namespace po = boost::program_options;
  po::options_description desc("Generate training data from perfect solver");
  desc.add_options()
      ("help,h", "help")
      ("c4-solver-dir,d", po::value<std::string>(&args.c4_solver_dir_str)->default_value(default_c4_solver_dir_str), "base dir containing c4solver bin and 7x6 book")
      ("num-mcts-iters,m", po::value<int>(&args.num_mcts_iters)->default_value(100), "num mcts iterations to do per move")
      ("num-search-threads,s", po::value<int>(&args.num_search_threads)->default_value(8), "num mcts search threads")
      ("batch-size-limit,b", po::value<int>(&args.batch_size_limit)->default_value(Mcts::kDefaultBatchSize),
          "num mcts search threads")
      ("num-games,g", po::value<int>(&args.num_games)->default_value(100), "num games to simulate")
#ifdef PROFILE_MCTS
      ("mcts-profiling-dir,p", po::value<std::string>(&args.mcts_profiling_dir)->default_value(
          Mcts::default_profiling_dir().string()), "directory in which to dump mcts profiling stats")
#endif  // PROFILE_MCTS
      ;

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (Mcts::kEnableProfiling) {
    if (args.mcts_profiling_dir.empty()) {
      throw util::Exception("Required: -p. Alternatively, add entry for 'mcts_profiling_dir' in config.txt");
    } else {
      Mcts::set_profiling_dir(args.mcts_profiling_dir);
    }
  }

  using player_t = common::AbstractPlayer<c4::GameState>;
  using player_array_t = std::array<player_t*, c4::kNumPlayers>;

  C4NNetPlayer* mcts_player = create_nnet_player(args);
  c4::PerfectPlayer* perfect_player = create_perfect_player(args);

  player_array_t players;
  players[c4::kRed] = mcts_player;
  players[c4::kYellow] = perfect_player;

  int win = 0;
  int loss = 0;
  int draw = 0;

  for (int i = 0; i < args.num_games; ++i) {
    common::GameRunner<c4::GameState> runner(players);
    auto outcome = runner.run();
    if (outcome[c4::kRed] == 1) {
      win++;
    } else if (outcome[c4::kYellow]) {
      loss++;
    } else {
      draw++;
    }

    printf("W%d L%d D%d", win, loss, draw);
    std::cout << std::endl;
  }
  return 0;
}
