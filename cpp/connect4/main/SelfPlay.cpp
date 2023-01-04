/*
 * TODO: replace all hard-coded c4 classes/functions here with macro values. Then, have the build.py process accept
 * a python file that specifies the game name, along with other relevant metadata (like filesystem paths to the relevant
 * c++ code). The build.py process can then import that file, and pass the macro values to cmake, ultimately causing
 * this file to get compiled to a game-specific binary.
 */
#include <array>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
  int parallelism_factor;
  bool verbose;
  bool perfect;
};

using Player = common::AbstractPlayer<c4::GameState>;
using C4NNetPlayer = common::NNetPlayer<c4::GameState, c4::Tensorizor>;
using Mcts = common::Mcts_<c4::GameState, c4::Tensorizor>;
using player_array_t = std::array<Player*, c4::kNumPlayers>;
using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
using duration_t = std::chrono::nanoseconds;
using win_loss_draw_array_t = std::array<int, 3>;

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

class SharedSelfPlayData {
public:
  bool request_game(int num_games) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (num_games_started_ >= num_games) return false;
    num_games_started_++;
    return true;
  }

  template<typename T> auto update(const T& outcome, int64_t ns) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (outcome[c4::kRed] == 1) {
      win_loss_draw_[0]++;
    } else if (outcome[c4::kYellow] == 1) {
      win_loss_draw_[1]++;
    } else {
      win_loss_draw_[2]++;
    }

    total_ns_ += ns;
    min_ns_ = std::min(min_ns_, ns);
    max_ns_ = std::max(max_ns_, ns);
    return win_loss_draw_;
  }

private:
  std::mutex mutex_;
  int num_games_started_ = 0;

  win_loss_draw_array_t win_loss_draw_ = {};
  int64_t total_ns_ = 0;
  int64_t min_ns_ = std::numeric_limits<int64_t>::max();
  int64_t max_ns_ = 0;
};

class SelfPlayThread {
public:
  SelfPlayThread(const Args& args, SharedSelfPlayData& shared_data)
  : args_(args)
  , shared_data_(shared_data)
  {
    p1_ = create_nnet_player(args_);
    p2_ = args_.perfect ? (Player*) create_perfect_player(args_) : (Player*) create_nnet_player(args_);

    players_[c4::kRed] = p1_;
    players_[c4::kYellow] = p2_;
  }

  ~SelfPlayThread() {
    if (thread_) delete thread_;
    delete p1_;
    delete p2_;
  }
  void join() { if (thread_ && thread_->joinable()) thread_->join(); }
  void launch() { thread_ = new std::thread([&] { run(); }); }

private:
  void run() {
    while (true) {
      if (!shared_data_.request_game(args_.num_games)) return;
      play_game();
    }
  }

  void play_game() {
    common::GameRunner<c4::GameState> runner(players_);
    time_point_t t1 = std::chrono::steady_clock::now();
    auto outcome = runner.run();
    time_point_t t2 = std::chrono::steady_clock::now();
    duration_t duration = t2 - t1;
    int64_t ns = duration.count();
    auto cumulative_win_loss_draw = shared_data_.update(outcome, ns);

    int cache_hits;
    int cache_misses;
    int cache_size;
    float hash_balance_factor;
    p1_->get_cache_stats(cache_hits, cache_misses, cache_size, hash_balance_factor);
    float cache_hit_rate = cache_hits * 1.0 / std::max(1, cache_hits + cache_misses);
    double ms = ns * 1e-6;

    int win = cumulative_win_loss_draw[0];
    int loss = cumulative_win_loss_draw[1];
    int draw = cumulative_win_loss_draw[2];

    printf("W%d L%d D%d | cache:[%.2f%% %d %.f] | %.3fms\n", win, loss, draw, 100 * cache_hit_rate,
           cache_size, hash_balance_factor, ms);
    std::flush(std::cout);
  }

  const Args& args_;
  SharedSelfPlayData& shared_data_;
  C4NNetPlayer* p1_;
  Player* p2_;
  player_array_t players_;

  std::thread* thread_ = nullptr;
};

class SelfPlay {
public:
  SelfPlay(const Args& args) : args_(args) {}

  void run() {
    int parallelism_factor = args_.parallelism_factor;
    for (int p = 0; p < parallelism_factor; ++p) {
      threads_.push_back(new SelfPlayThread(args_, shared_data_));
    }

    time_point_t t1 = std::chrono::steady_clock::now();

    for (auto thread : threads_) {
      thread->launch();
    }

    for (auto thread : threads_) {
      thread->join();
    }

    time_point_t t2 = std::chrono::steady_clock::now();
    duration_t duration = t2 - t1;
    int64_t ns = duration.count();

    printf("\nSelf-play complete!\n");
    printf("Parallelism factor:  %6d\n", parallelism_factor);
    printf("Num games:           %6d\n", args_.num_games);
    printf("MCTS iters:          %6d\n", args_.num_mcts_iters);
    printf("MCTS batch size:     %6d\n", Mcts::global_params_.batch_size_limit);
    printf("MCTS search threads: %6d\n", Mcts::global_params_.num_search_threads);
    printf("Total runtime:  %10.3fs\n", ns*1e-9);
    printf("Avg runtime:    %10.3fs\n", ns*1e-9 / args_.num_games);

    for (auto thread: threads_) {
      delete thread;
    }
  }

private:
  const Args& args_;
  std::vector<SelfPlayThread*> threads_;
  SharedSelfPlayData shared_data_;
};

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
      ("parallelism-factor,P", po::value<int>(&args.parallelism_factor)->default_value(1), "num games to play in parallel")
      ("verbose,v", po::bool_switch(&args.verbose)->default_value(false), "verbose")
      ("perfect,p", po::bool_switch(&args.perfect)->default_value(false), "play against perfect")
      ;

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  SelfPlay self_play(args);
  self_play.run();

  return 0;
}
