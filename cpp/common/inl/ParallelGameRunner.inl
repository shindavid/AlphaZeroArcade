#include <common/ParallelGameRunner.hpp>

#include <csignal>
#include <iostream>

#include <util/BoostUtil.hpp>
#include <util/StringUtil.hpp>

namespace common {

template<GameStateConcept GameState>
typename ParallelGameRunner<GameState>::runner_vec_t ParallelGameRunner<GameState>::active_runners;

template<GameStateConcept GameState>
auto ParallelGameRunner<GameState>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("ParallelGameRunner options");
  return desc
      .template add_option<"num-games", 'G'>(po::value<int>(&num_games)->default_value(num_games),
          "num games (<=0 means run indefinitely)")
      .template add_option<"parallelism", 'p'>(po::value<int>(&parallelism)->default_value(parallelism),
          "num games to play simultaneously")
      ;
}

template<GameStateConcept GameState>
ParallelGameRunner<GameState>::SharedData::SharedData(const Params& params) {
  if (params.display_progress_bar && params.num_games > 0) {
    bar_ = new progressbar(params.num_games + 1);  // + 1 for first update
    bar_->show_bar();  // so that progress-bar displays immediately
  }
}

template<GameStateConcept GameState>
bool ParallelGameRunner<GameState>::SharedData::request_game(int num_games) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (num_games > 0 && num_games_started_ >= num_games) return false;
  num_games_started_++;
  return true;
}

template<GameStateConcept GameState>
void ParallelGameRunner<GameState>::SharedData::update(const GameOutcome& outcome, int64_t ns) {
  std::lock_guard<std::mutex> guard(mutex_);
  for (player_index_t p = 0; p < kNumPlayers; ++p) {
    results_array_[p][outcome[p]]++;
  }

  total_ns_ += ns;
  min_ns_ = std::min(min_ns_, ns);
  max_ns_ = std::max(max_ns_, ns);
  if (bar_) bar_->update();
}

template<GameStateConcept GameState>
auto ParallelGameRunner<GameState>::SharedData::get_results() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return results_array_;
}

template<GameStateConcept GameState>
ParallelGameRunner<GameState>::GameThread::GameThread(const player_array_generator_t& gen, SharedData& shared_data)
: shared_data_(shared_data)
, players_(gen()) {}

template<GameStateConcept GameState>
ParallelGameRunner<GameState>::GameThread::~GameThread() {
  if (thread_) delete thread_;
  for (auto p : players_) delete p;
}

template<GameStateConcept GameState>
void ParallelGameRunner<GameState>::GameThread::run(const Params& params) {
  while (!shared_data_.terminated()) {
    if (!shared_data_.request_game(params.num_games)) return;
    play_game(params);
  }
}

template<GameStateConcept GameState>
void ParallelGameRunner<GameState>::GameThread::play_game(const Params& params) {
  bool print_result = !params.display_progress_bar;
  GameRunner runner(players_);
  time_point_t t1 = std::chrono::steady_clock::now();
  auto order = params.randomize_player_order ? GameRunner::kRandomPlayerSeats : GameRunner::kFixedPlayerSeats;
  auto outcome = runner.run(order);
  time_point_t t2 = std::chrono::steady_clock::now();
  duration_t duration = t2 - t1;
  int64_t ns = duration.count();
  shared_data_.update(outcome, ns);

  if (!print_result) return;

  results_array_t results = shared_data_.get_results();

  for (player_index_t p = 0; p < kNumPlayers; ++p) {
    printf("P%d %s | ", p, get_results_str(results[p]).c_str());
  }

  double ms = ns * 1e-6;
  printf("%.3fms\n", ms);
  std::flush(std::cout);
}

template<GameStateConcept GameState>
void ParallelGameRunner<GameState>::register_signal(int signum) {
  signal(signum, signal_handler);
}

template<GameStateConcept GameState>
void ParallelGameRunner<GameState>::signal_handler(int signum) {
  std::cout << "Caught signal! Gracefully terminating running games..." << std::endl;
  for (ParallelGameRunner* runner : active_runners) {
    runner->terminate();
  }
}

template<GameStateConcept GameState>
void ParallelGameRunner<GameState>::run() {
  active_runners.push_back(this);

  int parallelism = params_.parallelism;
  for (int p = 0; p < parallelism; ++p) {
    threads_.push_back(new GameThread(player_array_generator_, shared_data_));
  }

  time_point_t t1 = std::chrono::steady_clock::now();

  for (auto thread : threads_) {
    thread->launch(params_);
  }

  for (auto thread : threads_) {
    thread->join();
  }

  int num_games = shared_data_.num_games_started();
  time_point_t t2 = std::chrono::steady_clock::now();
  duration_t duration = t2 - t1;
  int64_t ns = duration.count();

  results_array_t results = shared_data_.get_results();

  printf("\nAll games complete!\n");
  for (player_index_t p = 0; p < kNumPlayers; ++p) {
    printf("P%d %s\n", p, get_results_str(results[p]).c_str());
  }
  PARAM_DUMP("Parallelism factor", "%d", parallelism);
  PARAM_DUMP("Num games", "%d", num_games);
  PARAM_DUMP("Total runtime", "%.3fs", ns*1e-9);
  PARAM_DUMP("Avg runtime", "%.3fs", ns*1e-9 / num_games);

  for (auto thread: threads_) {
    delete thread;
  }
}

template<GameStateConcept GameState>
std::string ParallelGameRunner<GameState>::get_results_str(const results_map_t& map) {
  int win = 0;
  int loss = 0;
  int draw = 0;
  float score = 0;

  for (auto it : map) {
    float f = it.first;
    int count = it.second;
    score += f * count;
    if (f == 1) win += count;
    else if (f == 0) loss += count;
    else draw += count;
  }
  return util::create_string("W%d L%d D%d [%.16g]", win, loss, draw, score);
}

}  // namespace common
