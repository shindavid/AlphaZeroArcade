#include <chrono>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

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
  int num_games;
  bool verbose;
};

using C4NNetPlayer = common::NNetPlayer<c4::GameState, c4::Tensorizor>;
using Mcts = common::Mcts_<c4::GameState, c4::Tensorizor>;

C4NNetPlayer* create_nnet_player(const Args& args) {
  C4NNetPlayer::Params params;
  params.num_mcts_iters = args.num_mcts_iters;
  params.temperature = 0;
  params.verbose = args.verbose;
  auto player = new C4NNetPlayer(params);
  player->set_name(util::create_string("MCTS-m%d", args.num_mcts_iters));
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
  po::options_description desc("Pit Mcts as red against perfect as yellow");
  desc.add_options()("help,h", "help");

  Mcts::global_params_.dirichlet_mult = 0;
  Mcts::add_options(desc);

  desc.add_options()
      ("c4-solver-dir,d", po::value<std::string>(&args.c4_solver_dir_str)->default_value(default_c4_solver_dir_str), "base dir containing c4solver bin and 7x6 book")
      ("num-mcts-iters,m", po::value<int>(&args.num_mcts_iters)->default_value(100), "num mcts iterations to do per move")
      ("num-games,g", po::value<int>(&args.num_games)->default_value(100), "num games to simulate")
      ("verbose,v", po::bool_switch(&args.verbose)->default_value(false), "verbose")
      ;

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
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

  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
  using duration_t = std::chrono::nanoseconds;
  int64_t total_ns = 0;
  int64_t min_ns = std::numeric_limits<int64_t>::max();
  int64_t max_ns = 0;

  int last_cache_hits = 0;
  int last_cache_misses = 0;

  for (int i = 0; i < args.num_games; ++i) {
    common::GameRunner<c4::GameState> runner(players);
    time_point_t t1 = std::chrono::steady_clock::now();
    auto outcome = runner.run();
    time_point_t t2 = std::chrono::steady_clock::now();
    if (outcome[c4::kRed] == 1) {
      win++;
    } else if (outcome[c4::kYellow] == 1) {
      loss++;
    } else {
      draw++;
    }

    duration_t duration = t2 - t1;
    int64_t ns = duration.count();
    total_ns += ns;
    min_ns = std::min(min_ns, ns);
    max_ns = std::max(max_ns, ns);

    int cache_hits;
    int cache_misses;
    int cache_size;
    float hash_balance_factor;
    mcts_player->get_cache_stats(cache_hits, cache_misses, cache_size, hash_balance_factor);
    int cur_cache_hits = cache_hits - last_cache_hits;
    int cur_cache_misses = cache_misses - last_cache_misses;
    float cache_hit_rate = cur_cache_hits * 1.0 / std::max(1, cur_cache_hits + cur_cache_misses);
    last_cache_hits = cache_hits;
    last_cache_misses = cache_misses;
    int wasted_evals = cache_misses - cache_size;  // assumes cache large enough that no evictions
    double ms = ns * 1e-6;

    printf("W%d L%d D%d | cache:[%.2f%% %d %d %.3f] | %.3fms", win, loss, draw, 100 * cache_hit_rate,
           wasted_evals, cache_size, hash_balance_factor, ms);
    std::cout << std::endl;
  }

  printf("Avg runtime: %.3fs\n", 1e-9 * total_ns / args.num_games);
  printf("Min runtime: %.3fs\n", 1e-9 * min_ns);
  printf("Max runtime: %.3fs\n", 1e-9 * max_ns);

  delete mcts_player;
  delete perfect_player;
  return 0;
}
