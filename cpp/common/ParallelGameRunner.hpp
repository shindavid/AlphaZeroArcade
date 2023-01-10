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
    int num_games = 1000;  // if <=0, run indefinitely
    int parallelism_factor = 100;  // number of games to run simultaneously
    bool display_progress_bar = false;
  };

  static Params global_params_;
  static void add_options(boost::program_options::options_description& desc, bool add_shortcuts=false);

  ParallelGameRunner(const Params& params) : params_(params), shared_data_(params_) {}
  ParallelGameRunner() : ParallelGameRunner(global_params_) {}

  void register_players(const player_array_generator_t& gen) { player_array_generator_ = gen; }
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

  private:
    mutable std::mutex mutex_;
    progressbar* bar_ = nullptr;
    int num_games_started_ = 0;

    results_array_t results_array_;
    int64_t total_ns_ = 0;
    int64_t min_ns_ = std::numeric_limits<int64_t>::max();
    int64_t max_ns_ = 0;
  };

  class GameThread {
  public:
    GameThread(const player_array_generator_t& gen, SharedData& shared_data);
    ~GameThread();

    void join() { if (thread_ && thread_->joinable()) thread_->join(); }
    void launch(int num_games) { thread_ = new std::thread([&] { run(num_games); }); }

  private:
    void run(int num_games);
    void play_game();

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
