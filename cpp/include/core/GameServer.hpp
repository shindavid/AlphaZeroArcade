#pragma once

#include <array>
#include <chrono>
#include <map>
#include <vector>

#include <core/AbstractPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/players/RemotePlayerProxyGenerator.hpp>
#include <core/TrainingDataWriter.hpp>
#include <third_party/ProgressBar.hpp>

namespace core {

template <concepts::Game Game>
class GameServer {
 public:
  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  using TrainingDataWriter = core::TrainingDataWriter<Game>;
  using TrainingDataWriterParams = TrainingDataWriter::Params;
  using GameWriteLog = TrainingDataWriter::GameWriteLog;
  using GameWriteLog_sptr = TrainingDataWriter::GameWriteLog_sptr;
  using GameResults = Game::GameResults;
  using ValueTensor = Game::Types::ValueTensor;
  using ValueArray = Game::Types::ValueArray;
  using ActionMask = Game::Types::ActionMask;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using TrainingInfo = Game::Types::TrainingInfo;
  using State = Game::State;
  using ChanceDistribution = Game::Types::ChanceDistribution;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using StateHistory = Game::StateHistory;
  using Rules = Game::Rules;
  using Player = AbstractPlayer<Game>;
  using PlayerGenerator = AbstractPlayerGenerator<Game>;
  using RemotePlayerProxyGenerator = core::RemotePlayerProxyGenerator<Game>;
  using player_array_t = std::array<Player*, kNumPlayers>;
  using player_name_array_t = Game::Types::player_name_array_t;
  using results_map_t = std::map<float, int>;
  using results_array_t = std::array<results_map_t, kNumPlayers>;
  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
  using duration_t = std::chrono::nanoseconds;
  using player_id_array_t = std::array<player_id_t, kNumPlayers>;
  using seat_index_array_t = std::array<seat_index_t, kNumPlayers>;

  /*
   * A PlayerInstantiation is instantiated from a PlayerRegistration. See PlayerRegistration for
   * more detail.
   */
  struct PlayerInstantiation {
    Player* player = nullptr;
    seat_index_t seat = -1;      // -1 means random seat
    player_id_t player_id = -1;  // order in which player was registered
  };
  using player_instantiation_array_t = std::array<PlayerInstantiation, kNumPlayers>;

  /*
   * A PlayerRegistration gives birth to a PlayerInstantiation.
   *
   * The difference is that a PlayerRegistration has a player-*generating-function*, rather than a
   * player. This is needed because when multiple GameThread's are launched, each needs to
   * instantiate its own Player. This requires the GameServer API to demand passing in a
   * player-*generator*, as opposed to a player, so that each spawned GameThread can create its own
   * player.
   */
  struct PlayerRegistration {
    PlayerGenerator* gen = nullptr;
    seat_index_t seat = -1;      // -1 means random seat
    player_id_t player_id = -1;  // order in which player was generated

    PlayerInstantiation instantiate(game_thread_id_t id) const {
      return {gen->generate_with_name(id), seat, player_id};
    }
  };
  using registration_vec_t = std::vector<PlayerRegistration>;

  struct Params {
    auto make_options_description();

    int num_games = 1000;   // if <=0, run indefinitely
    int parallelism = 256;  // number of games to run simultaneously
    int port = 0;
    float mean_noisy_moves = 0.0;  // mean of exp distr from which to draw number of noisy moves
    bool display_progress_bar = false;
    bool print_game_states = false;  // print game state between moves
    bool announce_game_results = false;  // print outcome of each individual match
    bool respect_victory_hints = false;  // quit game early if a player claims imminent victory
  };

 protected:
  static std::string get_results_str(const results_map_t& map);

 private:
  /*
   * Members of GameServer that need to be accessed by the individual GameThread's.
   */
  class SharedData {
   public:
    SharedData(const Params&, const TrainingDataWriterParams&);
    ~SharedData();

    const Params& params() const { return params_; }
    void init_progress_bar();
    bool request_game(int num_games);  // returns false iff hit num_games limit
    void update(const ValueArray& outcome, int64_t ns);
    auto get_results() const;
    void end_session();
    bool ready_to_start() const;
    int compute_parallelism_factor() const;
    int num_games_started() const { return num_games_started_; }
    void register_player(seat_index_t seat, PlayerGenerator* gen, bool implicit_remote = false);
    int num_registrations() const { return registrations_.size(); }
    player_instantiation_array_t generate_player_order(
        const player_instantiation_array_t& instantiations);
    void init_random_seat_indices();
    registration_vec_t& registration_templates() { return registrations_; }
    const std::string& get_player_name(player_id_t p) const {
      return registrations_[p].gen->get_name();
    }
    TrainingDataWriter* training_data_writer() const { return training_data_writer_; }

   private:
    const Params params_;

    TrainingDataWriter* training_data_writer_ = nullptr;

    registration_vec_t registrations_;
    seat_index_array_t random_seat_indices_;  // seats that will be assigned randomly
    int num_random_seats_ = 0;

    mutable std::mutex mutex_;
    progressbar* bar_ = nullptr;
    int num_games_started_ = 0;

    results_array_t results_array_;  // indexed by player_id
    int64_t total_ns_ = 0;
    int64_t min_ns_ = std::numeric_limits<int64_t>::max();
    int64_t max_ns_ = 0;
  };

  class GameThread {
   public:
    GameThread(SharedData& shared_data, game_thread_id_t);
    ~GameThread();

    void join() {
      if (thread_ && thread_->joinable()) thread_->join();
    }
    void launch() {
      thread_ = new std::thread([&] { run(); });
    }
    void decommission() { decommissioned_ = true; }

   private:
    void run();
    ValueArray play_game(player_array_t&);

    SharedData& shared_data_;
    player_instantiation_array_t instantiations_;
    std::thread* thread_ = nullptr;
    game_thread_id_t id_;
    bool decommissioned_ = false;
  };

 public:
  GameServer(const Params&, const TrainingDataWriterParams&);

  /*
   * A negative seat implies a random seat. Otherwise, the player generated is assigned the
   * specified seat.
   *
   * The player generator is assigned a unique player_id_t (0, 1, 2, ...), according to the order in
   * which the registrations are made. When aggregate game outcome stats are reported, they are
   * aggregated by player_id_t.
   *
   * Takes ownership of the pointer.
   */
  void register_player(seat_index_t seat, PlayerGenerator* gen) {
    shared_data_.register_player(seat, gen);
  }

  /*
   * Blocks until all players have registered.
   */
  void wait_for_remote_player_registrations();

  const Params& params() const { return shared_data_.params(); }
  int get_port() const { return params().port; }
  int num_registered_players() const { return shared_data_.num_registrations(); }
  void run();
  void shutdown();

 private:
  SharedData shared_data_;
  std::vector<GameThread*> threads_;
};

}  // namespace core

#include <inline/core/GameServer.inl>
