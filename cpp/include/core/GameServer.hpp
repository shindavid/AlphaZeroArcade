#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/AbstractPlayerGenerator.hpp"
#include "core/ActionRequest.hpp"
#include "core/BacktrackUpdate.hpp"
#include "core/BasicTypes.hpp"
#include "core/ChanceEventHandleRequest.hpp"
#include "core/GameServerBase.hpp"
#include "core/GameStateTree.hpp"
#include "core/LoopControllerListener.hpp"
#include "core/PerfStats.hpp"
#include "core/StateChangeUpdate.hpp"
#include "core/YieldManager.hpp"
#include "core/concepts/GameConcept.hpp"
#include "core/players/RemotePlayerProxyGenerator.hpp"
#include "third_party/ProgressBar.hpp"
#include "util/CompactBitSet.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
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

  enum pause_state_t : int8_t { kPausing, kPaused, kUnpausing, kUnpaused };

  using enqueue_instruction_t = core::GameServerBase::enqueue_instruction_t;
  using next_result_t = core::GameServerBase::next_result_t;
  using EnqueueRequest = core::GameServerBase::EnqueueRequest;
  using StepResult = core::GameServerBase::StepResult;
  using CriticalSectionCheck = core::GameServerBase::CriticalSectionCheck;

  using GameResults = Game::GameResults;
  using GameResultTensor = Game::Types::GameResultTensor;
  using ValueArray = Game::Types::ValueArray;
  using ActionMask = Game::Types::ActionMask;
  using ChanceEventHandleRequest = core::ChanceEventHandleRequest<Game>;
  using ActionRequest = core::ActionRequest<Game>;
  using State = Game::State;
  using ChanceDistribution = Game::Types::ChanceDistribution;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using StateChangeUpdate = core::StateChangeUpdate<Game>;
  using BacktrackUpdate = core::BacktrackUpdate<Game>;
  using ReverseHistory = core::ReverseHistory<State>;
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
  using action_vec_t = std::vector<action_t>;
  using StateTree = GameStateTree<Game>;
  using BacktrackingSupport = util::CompactBitSet<kNumPlayers>;

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

    std::string initial_actions_str;  // integers separated by commas
    int num_games = 1024;             // if <=0, run indefinitely
    int parallelism = 1024;           // number of games to run simultaneously
    int num_game_threads = 16;        // number of threads to use
    int port = 0;
    float mean_noisy_moves = 0.0;  // mean of exp distr from which to draw number of noisy moves
    bool display_progress_bar = false;
    bool print_game_states = false;      // print game state between moves
    bool announce_game_results = false;  // print outcome of each individual match
    bool respect_victory_hints = true;   // quit game early if a player claims imminent victory
    bool analysis_mode = false;          // external controller steps through the game

    // The game server can choose to alternate between players, like so:
    //
    // 1. All instances of player 0 make their moves
    // 2. All instances of player 1 make their moves
    // 3. All instances of player 0 make their moves
    // 4. All instances of player 1 make their moves
    // ...
    //
    // This can improve performance in settings where we have multiple players using different
    // neural network models, but sharing the same physical GPU. Alternation ensures that the
    // multiple NNEvaluationService's don't clash with each other.
    //
    // There are 3 valid settings to control this behavior:
    //
    // 0: Explicitly disable alternation.
    // 1: Auto-enable alternation, based on the registered players (default).
    // 2. Explicitly enable alternation.
    //
    // Auto-enable alternation means that alternation is enabled if there are multiple
    // NNEvaluationService's registered, and disabled otherwise. This is the default.
    //
    // Note that in settings where remote players are used, sharing the same physical GPU, we will
    // want to enable alternation, but GameServer cannot detect this automatically. It is for this
    // reason that we provide the option to explicitly enable alternation.
    int alternating_mode = 1;
  };

 protected:
  static std::string get_results_str(const results_map_t& map);

 private:
  class SharedData;  // forward declaration
  class GameSlot {
   public:
    GameSlot(SharedData&, game_slot_index_t);
    ~GameSlot();

    StepResult step(context_id_t context);

    bool start_game();
    bool game_started() const { return game_started_; }
    bool game_ended() const { return !game_started_; }
    game_slot_index_t id() const { return id_; }
    player_id_t active_player_id() const { return player_order_[active_seat_].player_id; }
    seat_index_t active_seat() const { return active_seat_; }
    Player* active_player() const { return active_seat_ < 0 ? nullptr : players_[active_seat_]; }

    bool mid_yield() const { return mid_yield_; }
    bool in_critical_section() const { return in_critical_section_; }
    const State& state() const { return state_tree_.state(state_node_index_); }
    void apply_action(action_t action);

   private:
    const Params& params() const { return shared_data_.params(); }
    void pre_step();

    // Returns true if it successfully processed a non-terminal game state transition.
    bool step_chance(StepResult& result);

    // Returns true if it successfully processed a non-terminal game state transition. Also sets
    // request to the appropriate value.
    bool step_non_chance(context_id_t context, StepResult& result);

    void handle_terminal(const GameResultTensor& outcome, StepResult& result);

    game_tree_node_aux_t get_player_aux() const {
      return state_tree_.get_player_aux(state_node_index_, active_seat_);
    }

    void set_player_aux(game_tree_node_aux_t aux) {
      state_tree_.set_player_aux(state_node_index_, active_seat_, aux);
    }

    void backtrack_to_node(game_tree_index_t backtrack_node_ix);
    game_tree_index_t player_last_action_node_index() const;
    bool active_player_supports_backtracking() const;

    bool undo_allowed() const { return state_tree_.player_acted(state_node_index_, active_seat_); }
    void undo_player_last_action() { state_node_index_ = player_last_action_node_index(); }
    void resign_game(StepResult& result);

    SharedData& shared_data_;
    const game_slot_index_t id_;
    player_instantiation_array_t instantiations_;

    // Initialized at the start of the game
    game_id_t game_id_;
    player_instantiation_array_t player_order_;
    player_array_t players_;
    player_name_array_t player_names_;
    int num_noisy_starting_moves_ = 0;
    bool game_started_ = false;

    // Updated for each move
    StateTree state_tree_;
    game_tree_index_t state_node_index_ = kNullNodeIx;
    ActionMask valid_actions_;
    int move_number_;  // tracks player-actions, not chance-events
    int step_chance_player_index_ = 0;
    action_t chance_action_ = kNullAction;
    core::action_mode_t action_mode_;
    seat_index_t active_seat_;
    bool noisy_mode_;
    bool mid_yield_;

    // Defensive programming
    std::atomic<bool> in_critical_section_ = false;
  };

  /*
   * Members of GameServer that need to be accessed by the individual GameSlot's.
   */
  class SharedData {
   public:
    SharedData(GameServer* server, const Params&);
    ~SharedData();

    const Params& params() const { return params_; }

    void init_slots();
    void start_games();
    void init_progress_bar();
    void init_random_seat_indices();

    int num_slots() const { return game_slots_.size(); }

    // Gets the next item from the queue. Sets:
    //
    // - item: with the next queue item
    // - wait_for_game_slot_time_ns: with the time spent waiting
    next_result_t next(int64_t& wait_for_game_slot_time_ns, SlotContext& item);
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
    int num_games_ended() const { return num_games_ended_; }
    int num_registrations() const { return registrations_.size(); }
    registration_vec_t& registration_templates() { return registrations_; }
    YieldManager* yield_manager() { return &yield_manager_; }
    pause_state_t pause_state() const { return pause_state_; }

    void handle_alternating_mode_recommendation();
    void debug_dump() const;
    void pause();
    void unpause();
    void run_prelude(core::game_thread_id_t id);
    void increment_active_thread_count();
    void decrement_active_thread_count();
    void set_num_initial_threads(int n) { num_initial_threads_ = n; }

    void increment_mcts_time_ns(int64_t ns) { mcts_time_ns_ += ns; }
    void increment_game_slot_time_ns(int64_t ns) { wait_for_game_slot_time_ns_ += ns; }
    void update_perf_stats(PerfStats&);
    const action_vec_t& initial_actions() const { return server_->initial_actions(); }
    const BacktrackingSupport& backtracking_support() const { return backtracking_support_; }

   private:
    void state_loop();
    slot_context_queue_t& get_queue_to_use(game_slot_index_t);
    void increment_global_active_player_id() {
      global_active_player_id_ = (global_active_player_id_ + 1) % kNumPlayers;
    }
    void validate_deferred_count() const;  // for debugging

    GameServer* const server_;
    const Params params_;

    registration_vec_t registrations_;
    seat_index_array_t random_seat_indices_;  // seats that will be assigned randomly
    int num_random_seats_ = 0;
    BacktrackingSupport backtracking_support_;

    mit::condition_variable cv_;
    mutable mit::mutex mutex_;
    mutable mit::mutex perf_stats_mutex_;
    progressbar* bar_ = nullptr;
    int num_games_started_ = 0;
    int num_games_ended_ = 0;

    // game_slots_ is in a fixed order, and doesn't change after initialization. This data
    // structure is used to look up the GameSlot for a given game_slot_index_t, which is needed for
    // efficient yield/notify mechanics.
    std::vector<GameSlot*> game_slots_;

    // We constantly pop from the front of the queue via next(), and push back to the end of the
    // queue via enqueue(). In between next() and enqueue(), pending_queue_count_ is incremented.
    //
    // During the normal course of operations, (queue_.size() + pending_queue_count_) should equal
    // game_slots_.size(). When a shutdown commences, queue_ will whittle down to zero.
    slot_context_queue_t queue_;

    // Used in alternating mode to house queue items that are deferred until it's the right
    // player's turn. Items in these queues are part of pending_queue_count_.
    slot_context_queue_t deferred_queues_[kNumPlayers];

    // The number of items that are awaiting notification. These items will be added to queue_ upon
    // notification. In alternating mode, this count includes deferred_count_.
    int pending_queue_count_ = 0;

    // The sum of the sizes of all deferred_queues_. Can only be non-zero in alternating mode.
    int deferred_count_ = 0;

    YieldManager yield_manager_;

    mit::thread state_thread_;

    results_array_t results_array_;  // indexed by player_id

    int num_initial_threads_ = 0;
    int active_thread_count_ = 0;
    int paused_thread_count_ = 0;
    int in_prelude_count_ = 0;
    player_id_t global_active_player_id_ = -1;  // used in alternating mode
    pause_state_t pause_state_ = kUnpaused;
    bool waiting_in_next_ = false;
    bool state_thread_launched_ = false;

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
    mit::thread thread_;
    game_thread_id_t id_;
  };

  virtual void handle_alternating_mode_recommendation() override;

  virtual void debug_dump() const override { shared_data_.debug_dump(); }

 public:
  GameServer(const Params&);

  void set_initial_actions(const action_vec_t& initial_actions) {
    initial_actions_ = initial_actions;
  }

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
  void setup();
  void run();
  void print_summary() const;
  void create_threads();
  void launch_threads();
  void join_threads();

  void pause() override { shared_data_.pause(); }
  void unpause() override { shared_data_.unpause(); }
  void update_perf_stats(PerfStats&) override;
  const action_vec_t& initial_actions() const { return initial_actions_; }
  SharedData& shared_data() { return shared_data_; }

 private:
  SharedData shared_data_;
  std::vector<GameThread*> threads_;
  action_vec_t initial_actions_;
};

}  // namespace core

#include "inline/core/GameServer.inl"
