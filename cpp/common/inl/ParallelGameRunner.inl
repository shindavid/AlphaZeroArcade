#include <common/ParallelGameRunner.hpp>

#include <iostream>

#include <util/StringUtil.hpp>

namespace common {

template<GameStateConcept GameState>
typename ParallelGameRunner<GameState>::Params ParallelGameRunner<GameState>::global_params_;

template<GameStateConcept GameState>
void ParallelGameRunner<GameState>::add_options(boost::program_options::options_description& desc, bool add_shortcuts) {
  namespace po = boost::program_options;

  Params& params = global_params_;
  desc.add_options()
      (add_shortcuts ? "num-games,g" : "num-games", po::value<int>(&params.num_games)->default_value(params.num_games),
       "num games (<=0 means run indefinitely)")
      (add_shortcuts ? "parallelism-factor,p" : "parallelism-factor",
          po::value<int>(&params.parallelism_factor)->default_value(params.parallelism_factor),
          "num games to play simultaneously")
      ;
}

template<GameStateConcept GameState>
ParallelGameRunner<GameState>::SharedData::SharedData(const Params& params) {
  if (params.display_progress_bar) {
    bar_ = new progressbar(params.num_games + 1);  // + 1 for first update
    bar_->show_bar();  // so that progress-bar displays immediately
  }
}

template<GameStateConcept GameState>
bool ParallelGameRunner<GameState>::SharedData::request_game(int num_games) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (num_games >= 0 && num_games_started_ >= num_games) return false;
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
  while (true) {
    if (!shared_data_.request_game(params.num_games)) return;
    play_game(!params.display_progress_bar);
  }
}

template<GameStateConcept GameState>
void ParallelGameRunner<GameState>::GameThread::play_game(bool print_result) {
  GameRunner runner(players_);
  time_point_t t1 = std::chrono::steady_clock::now();
  auto outcome = runner.run();
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
void ParallelGameRunner<GameState>::run() {
  int parallelism_factor = params_.parallelism_factor;
  for (int p = 0; p < parallelism_factor; ++p) {
    threads_.push_back(new GameThread(player_array_generator_, shared_data_));
  }

  time_point_t t1 = std::chrono::steady_clock::now();

  for (auto thread : threads_) {
    thread->launch(params_);
  }

  for (auto thread : threads_) {
    thread->join();
  }

  time_point_t t2 = std::chrono::steady_clock::now();
  duration_t duration = t2 - t1;
  int64_t ns = duration.count();

  results_array_t results = shared_data_.get_results();

  printf("\nAll games complete!\n");
  for (player_index_t p = 0; p < kNumPlayers; ++p) {
    printf("P%d %s\n", p, get_results_str(results[p]).c_str());
  }
  printf("Parallelism factor:  %6d\n", parallelism_factor);
  printf("Num games:           %6d\n", params_.num_games);
  printf("Total runtime:  %10.3fs\n", ns*1e-9);
  printf("Avg runtime:    %10.3fs\n", ns*1e-9 / params_.num_games);

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
  return util::create_string("W%d L%d D%d [%.3g]", win, loss, draw, score);
}

}  // namespace common
