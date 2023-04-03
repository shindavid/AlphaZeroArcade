#pragma once

#include <array>
#include <chrono>
#include <map>
#include <vector>

#include <common/AbstractPlayer.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/BasicTypes.hpp>
#include <third_party/ProgressBar.hpp>

namespace common {

template<GameStateConcept GameState>
class GameServer {
public:
  static constexpr int kNumPlayers = GameState::kNumPlayers;

  using GameStateTypes = common::GameStateTypes<GameState>;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using ActionMask = typename GameStateTypes::ActionMask;
  using Player = AbstractPlayer<GameState>;
  using player_array_t = std::array<Player*, kNumPlayers>;
  using player_name_array_t = typename GameStateTypes::player_name_array_t;
  using player_generator_t = std::function<Player*()>;
  using results_map_t = std::map<float, int>;
  using results_array_t = std::array<results_map_t, kNumPlayers>;
  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
  using duration_t = std::chrono::nanoseconds;
  using player_id_t = int;

  /*
   * A registration_t is instantiated from a registration_template_t. See registration_template_t for more detail.
   */
  struct registration_t {
    Player* player;
    player_index_t seat;  // -1 means random seat
    player_id_t player_id;  // order in which player was registered
  };
  using registration_array_t = std::array<registration_t, kNumPlayers>;

  /*
   * A registration_template_t gives birth to a registration_t.
   *
   * The difference is that a registration_template_t has a player-*generating-function*, rather than a player.
   * This is needed because when multiple GameThread's are launched, each needs to instantiate its own Player.
   * This requires the GameServer API to demand passing in a player-*generator*, as opposed to a player, so that
   * each spawned GameThread can create its own player.
   */
  struct registration_template_t {
    player_generator_t gen;
    player_index_t seat;  // -1 means random seat
    player_id_t player_id;  // order in which player was generated

    registration_t instantiate() const { return {gen(), seat, player_id}; }
  };
  using registration_template_array_t = std::array<registration_template_t, kNumPlayers>;

  struct Params {
    auto make_options_description();

    int num_games = 1000;  // if <=0, run indefinitely
    int parallelism = 100;  // number of games to run simultaneously
    int port = 0;
    bool display_progress_bar = false;
  };

protected:
  static std::string get_results_str(const results_map_t& map);

private:
  /*
   * Members of GameServer that need to be accessed by the individual GameThread's.
   */
  class SharedData {
  public:
    SharedData(const Params& params);
    ~SharedData() { if (bar_) delete bar_; }

    bool request_game(int num_games);  // returns false iff hit num_games limit
    void update(const GameOutcome& outcome, int64_t ns);
    auto get_results() const;
    int num_games_started() const { return num_games_started_; }
    player_id_t register_player(player_index_t seat, player_generator_t gen);
    int num_registrations() const { return num_registrations_; }
    registration_array_t generate_player_order(const registration_array_t& registrations) const;
    const registration_template_array_t& registration_templates() const { return registration_templates_; }

  private:
    registration_template_array_t registration_templates_;
    int num_registrations_ = 0;

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
    GameThread(SharedData& shared_data);
    ~GameThread();

    void join() { if (thread_ && thread_->joinable()) thread_->join(); }
    void launch(const Params&);

  private:
    void run(const Params&);
    GameOutcome play_game(const player_array_t&);

    SharedData& shared_data_;
    registration_array_t registrations_;
    std::thread* thread_ = nullptr;
  };

public:
  GameServer(const Params& params);

  /*
   * If seat is not specified, then the player generator is assigned a random seat.
   *
   * Otherwise, the player generated is assigned the specified seat.
   *
   * The player generator is assigned a unique player_id_t (0, 1, 2, ...), according to the order in which the
   * registrations are made. This value is returned by this function. When aggregate game outcome stats are reported,
   * they are aggregated by player_id_t.
   */
  player_id_t register_player(player_generator_t gen) { return register_player(-1, gen); }
  player_id_t register_player(player_index_t seat, player_generator_t gen) {
    return shared_data_.register_player(seat, gen);
  }

  /*
   * Blocks until all players have registered.
   */
  void wait_for_remote_player_registrations();

  int port() const { return params_.port; }
  int num_registered_players() const { return shared_data_.num_registrations(); }
  bool ready_to_start() const { return num_registered_players() == kNumPlayers; }
  void run();

private:
  const Params params_;
  std::vector<GameThread*> threads_;
  SharedData shared_data_;
};

}  // namespace common

#include <common/inl/GameServer.inl>
