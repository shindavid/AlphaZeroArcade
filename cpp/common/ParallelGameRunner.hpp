#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <common/AbstractPlayer.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameRunner.hpp>
#include <common/GameStateConcept.hpp>
#include <third_party/ProgressBar.hpp>
#include <util/BoostUtil.hpp>

namespace common {

template<GameStateConcept GameState>
class ParallelGameRunner {
public:
  static constexpr int kNumPlayers = GameState::kNumPlayers;

  using GameStateTypes = common::GameStateTypes<GameState>;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using Player = AbstractPlayer<GameState>;
  using GameRunner = common::GameRunner<GameState>;

  using player_array_t = std::array<Player*, kNumPlayers>;
  using player_array_generator_t = std::function<player_array_t()>;
  using results_map_t = std::map<float, int>;
  using results_array_t = std::array<results_map_t, kNumPlayers>;

  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
  using duration_t = std::chrono::nanoseconds;

  struct Params {
    auto make_options_description();

    int num_games = 1000;  // if <=0, run indefinitely
    int parallelism = 100;  // number of games to run simultaneously
    bool randomize_player_order = true;
    bool display_progress_bar = false;
  };

  using runner_vec_t = std::vector<ParallelGameRunner*>;
  static runner_vec_t active_runners;
  static void register_signal(int signum);
  static void signal_handler(int signum);

  ParallelGameRunner(const Params& params) : params_(params), shared_data_(params_) {}

  void register_players(const player_array_generator_t& gen) { player_array_generator_ = gen; }
  void terminate() { shared_data_.terminate(); }
  void run();

protected:
  static std::string get_results_str(const results_map_t& map);

private:
  class SharedData {
  public:
    SharedData(const Params& params);
    ~SharedData() { if (bar_) delete bar_; }

    bool request_game(int num_games);  // returns false iff hit num_games limit
    void update(const GameOutcome& outcome, int64_t ns);
    auto get_results() const;
    void terminate() { terminated_ = true; }
    bool terminated() const { return terminated_; }
    int num_games_started() const { return num_games_started_; }

  private:
    mutable std::mutex mutex_;
    progressbar* bar_ = nullptr;
    int num_games_started_ = 0;

    results_array_t results_array_;
    int64_t total_ns_ = 0;
    int64_t min_ns_ = std::numeric_limits<int64_t>::max();
    int64_t max_ns_ = 0;
    bool terminated_ = false;
  };

  class GameThread {
  public:
    GameThread(const player_array_generator_t& gen, SharedData& shared_data);
    ~GameThread();

    void join() { if (thread_ && thread_->joinable()) thread_->join(); }
    void launch(const Params& params) { thread_ = new std::thread([&] { run(params); }); }

  private:
    void run(const Params& params);
    void play_game(const Params& params);

    SharedData& shared_data_;
    player_array_t players_;
    std::thread* thread_ = nullptr;
  };

  Params params_;
  player_array_generator_t player_array_generator_;
  std::vector<GameThread*> threads_;
  SharedData shared_data_;
};

}  // namespace common

#include <common/inl/ParallelGameRunner.inl>
