#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/BasicTypes.hpp>
#include <core/GameServerBase.hpp>
#include <core/LoopControllerListener.hpp>
#include <core/PerfStats.hpp>
#include <core/TrainingDataWriter.hpp>
#include <core/YieldManager.hpp>
#include <core/concepts/Game.hpp>
#include <core/players/RemotePlayerProxyGenerator.hpp>
#include <third_party/ProgressBar.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <queue>
#include <vector>

namespace core {

// GameServer is used to run games, potentially in parallel, between players. It is responsible for
// managing the game state, handling player actions, and communicating with the players. It also
// provides a mechanism for training data collection and logging game results.
//
// The overall architecture is driven by the empirical observation that using too many threads can
// lead to performance degradation, particularly in cloud environments. In the past, we had one
// game-thread per active-game, which led to a tension: we want many active games to fully
// saturate the GPU for batch nn evaluation, but we also want to limit the number of threads to
// avoid overhead.
//
// ** The solution is to decouple the number of active games from the number of game threads. **
//
// In our new implementation, we have T GameThread's and P GameSlot's, with T << P. The GameSlot's
// will reside in a queue, and each GameThread will run in a loop, constantly pulling from the queue
// and processing the next GameSlot. When processing a GameSlot, it will keep working on that
// instance until the current player returns an ActionResponse. That ActionResponse might be a move
// in the game, or it might be a "yield", which means that the player is blocked (e.g., waiting for
// an nn evaluation to finish). In that case, the GameThread will skip to the next GameSlot in the
// queue.
//
// For MCTS self-play, the ideal configuration will be such that when a GameThread revisits a
// GameSlot that previously yielded, the nn evaluation will typically be finished, so that a
// GameThread never spends time waiting.
template <concepts::Game Game>
class GameServer
    : public core::PerfStatsClient,
      public core::GameServerBase,
      public core::LoopControllerListener<core::LoopControllerInteractionType::kPause> {
 public:
  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  using enqueue_instruction_t = core::GameServerBase::enqueue_instruction_t;
  using EnqueueRequest = core::GameServerBase::EnqueueRequest;

  using TrainingDataWriter = core::TrainingDataWriter<Game>;
  using TrainingDataWriterParams = TrainingDataWriter::Params;
  using GameWriteLog = TrainingDataWriter::GameWriteLog;
  using GameWriteLog_sptr = TrainingDataWriter::GameWriteLog_sptr;
  using GameResults = Game::GameResults;
  using ValueTensor = Game::Types::ValueTensor;
  using ValueArray = Game::Types::ValueArray;
  using ActionMask = Game::Types::ActionMask;
  using ChangeEventPreHandleRequest = Game::Types::ChangeEventPreHandleRequest;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using ChanceEventPreHandleResponse = Game::Types::ChanceEventPreHandleResponse;
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
   * player. This is needed because when multiple GameSlot's are created, each needs to
   * instantiate its own Player. This requires the GameServer API to demand passing in a
   * player-*generator*, as opposed to a player, so that each GameSlot can create its own player.
   */
  struct PlayerRegistration {
    PlayerGenerator* gen = nullptr;
    seat_index_t seat = -1;      // -1 means random seat
    player_id_t player_id = -1;  // order in which player was generated

    PlayerInstantiation instantiate(game_slot_index_t id) const {
      return {gen->generate_with_name(id), seat, player_id};
    }
  };
  using registration_vec_t = std::vector<PlayerRegistration>;

  struct Params {
    auto make_options_description();

    int num_games = 1024;       // if <=0, run indefinitely
    int parallelism = 1024;      // number of games to run simultaneously
    int num_game_threads = 16;  // number of threads to use
    int port = 0;
    float mean_noisy_moves = 0.0;  // mean of exp distr from which to draw number of noisy moves
    bool display_progress_bar = false;
    bool print_game_states = false;  // print game state between moves
    bool announce_game_results = false;  // print outcome of each individual match
    bool respect_victory_hints = true;  // quit game early if a player claims imminent victory
  };

 protected:
  static std::string get_results_str(const results_map_t& map);

 private:
  class SharedData;  // forward declaration

  class GameSlot {
   public:
    GameSlot(SharedData&, game_slot_index_t);
    ~GameSlot();

    EnqueueRequest step(context_id_t context);

    bool start_game();
    bool game_started() const { return game_started_; }
    bool game_ended() const { return !game_started_; }
    game_slot_index_t id() const { return id_; }

   private:
    const Params& params() const { return shared_data_.params(); }
    void pre_step();

    // Returns true if it successfully processed a non-terminal game state transition.
    bool step_chance();

    // Returns true if it successfully processed a non-terminal game state transition. Also sets
    // request to the appropriate value.
    bool step_non_chance(context_id_t context, EnqueueRequest& request);

    void handle_terminal(const ValueTensor& outcome);

    SharedData& shared_data_;
    const game_slot_index_t id_;
    player_instantiation_array_t instantiations_;

    // Initialized at the start of the game
    game_id_t game_id_;
    GameWriteLog_sptr game_log_;
    player_instantiation_array_t player_order_;
    player_array_t players_;
    player_name_array_t player_names_;
    int num_noisy_starting_moves_ = 0;
    bool game_started_ = false;

    // Updated for each move
    StateHistory state_history_;
    ActionMask valid_actions_;
    ActionValueTensor* chance_action_values_ = nullptr;
    int move_number_;  // tracks player-actions, not chance-events
    int step_chance_player_index_ = 0;
    core::action_mode_t action_mode_;
    seat_index_t active_seat_;
    bool noisy_mode_;
    bool mid_yield_;

    // Used for synchronization in multithreaded case
    std::atomic<int> pending_drop_count_ = 0;
  };

  /*
   * Members of GameServer that need to be accessed by the individual GameSlot's.
   */
  class SharedData {
   public:
    SharedData(GameServer* server, const Params&, const TrainingDataWriterParams&);
    ~SharedData();

    const Params& params() const { return params_; }

    void init_slots();
    void start_games();
    void init_progress_bar();
    void init_random_seat_indices();
    void run_yield_manager();

    int num_slots() const { return game_slots_.size(); }

    // If the server is paused or shutting down, returns false. Else, returns true, and sets:
    //
    // - item: with the next queue item
    // - wait_for_game_slot_time_ns: with the time spent waiting
    bool next(int64_t& wait_for_game_slot_time_ns, SlotContext& item);
    void enqueue(SlotContext, const EnqueueRequest& request);
    GameSlot* get_game_slot(game_slot_index_t id) { return game_slots_[id]; }
    void drop_slot();

    bool request_game();  // returns false iff hit params_.num_games limit
    void update(const ValueArray& outcome);
    auto get_results() const;
    void start_session();
    void end_session();
    bool ready_to_start() const;
    void register_player(seat_index_t seat, PlayerGenerator* gen, bool implicit_remote = false);
    player_instantiation_array_t generate_player_order(
      const player_instantiation_array_t& instantiations);

    const std::string& get_player_name(player_id_t p) const {
      return registrations_[p].gen->get_name();
    }
    int num_games_started() const { return num_games_started_; }
    int num_registrations() const { return registrations_.size(); }
    registration_vec_t& registration_templates() { return registrations_; }
    TrainingDataWriter* training_data_writer() const { return training_data_writer_; }
    YieldManager* yield_manager() { return &yield_manager_; }
    bool paused() const { return paused_; }

    void pause();
    void unpause();
    void wait_for_unpause();
    void increment_active_thread_count();
    void decrement_active_thread_count();
    void increment_paused_thread_count();
    void decrement_paused_thread_count();

    void increment_mcts_time_ns(int64_t ns) { mcts_time_ns_ += ns; }
    void increment_game_slot_time_ns(int64_t ns) { wait_for_game_slot_time_ns_ += ns; }
    void update_perf_stats(PerfStats&);

   private:
    bool queue_pending() const { return pending_queue_count_ > 0 && queue_.empty(); }
    void issue_pause_receipt_if_necessary();  // assumes mutex_ is locked

    GameServer* const server_;
    const Params params_;

    TrainingDataWriter* training_data_writer_ = nullptr;

    registration_vec_t registrations_;
    seat_index_array_t random_seat_indices_;  // seats that will be assigned randomly
    int num_random_seats_ = 0;

    std::condition_variable cv_;
    mutable std::mutex mutex_;
    mutable std::mutex perf_stats_mutex_;
    progressbar* bar_ = nullptr;
    int num_games_started_ = 0;

    // game_slots_ is in a fixed order, and doesn't change after initialization. This data
    // structure is used to look up the GameSlot for a given game_slot_index_t, which is needed for
    // efficient remote proxy mechanics.
    std::vector<GameSlot*> game_slots_;

    // We constantly pop from the front of the queue via next(), and push back to the end of the
    // queue via enqueue(). In between next() and enqueue(), pending_queue_count_ is incremented.
    //
    // During the normal course of operations, (queue_.size() + pending_queue_count_) should equal
    // game_slots_.size(). When a shutdown commences, queue_ will whittle down to zero.
    std::queue<SlotContext> queue_;
    int pending_queue_count_ = 0;

    YieldManager yield_manager_;

    results_array_t results_array_;  // indexed by player_id

    int active_thread_count_ = 0;
    int paused_thread_count_ = 0;
    bool paused_ = false;
    bool pause_receipt_pending_ = false;

    std::atomic<int64_t> mcts_time_ns_ = 0;
    std::atomic<int64_t> wait_for_game_slot_time_ns_ = 0;
  };

  class GameThread {
   public:
    GameThread(SharedData& shared_data, game_thread_id_t);
    ~GameThread();

    void join();
    void launch();

   private:
    void run();

    SharedData& shared_data_;
    std::thread thread_;
    game_thread_id_t id_;
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
  void create_threads();
  void launch_threads();
  void join_threads();

  void pause() override { shared_data_.pause(); }
  void unpause() override { shared_data_.unpause(); }
  void update_perf_stats(PerfStats&) override;

 private:
  SharedData shared_data_;
  std::vector<GameThread*> threads_;
};

}  // namespace core

#include <inline/core/GameServer.inl>
