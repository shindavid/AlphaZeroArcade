#pragma once

#include <core/AbstractPlayerGenerator.hpp>
#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/HibernationManager.hpp>
#include <core/HibernationNotifier.hpp>
#include <core/Packet.hpp>
#include <util/CppUtil.hpp>
#include <util/SocketUtil.hpp>

#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

namespace core {

template <concepts::Game Game>
class GameServerProxy {
 public:
  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr bool kEnableDebug = IS_MACRO_ENABLED(GAME_SERVER_PROXY_DEBUG);

  using State = Game::State;
  using StateHistory = Game::StateHistory;
  using Rules = Game::Rules;
  using ActionMask = Game::Types::ActionMask;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using ValueTensor = Game::Types::ValueTensor;
  using PlayerGenerator = AbstractPlayerGenerator<Game>;
  using player_generator_array_t = std::array<PlayerGenerator*, kNumPlayers>;
  using Player = AbstractPlayer<Game>;
  using player_name_array_t = Player::player_name_array_t;
  using player_array_t = std::array<Player*, kNumPlayers>;
  using player_vec_t = std::vector<Player*>;

  struct SeatGenerator {
    seat_index_t seat;
    PlayerGenerator* gen;
  };
  using seat_generator_vec_t = std::vector<SeatGenerator>;

  struct Params {
    auto make_options_description();

    std::string remote_server = "localhost";
    int remote_port = 0;
  };

  class SharedData;  // forward declaration

  class GameSlot {
   public:
    GameSlot(SharedData&, game_slot_index_t);
    ~GameSlot();

    void handle_start_game(const StartGame& payload);
    void handle_state_change(const StateChange& payload);
    void handle_action_prompt(const ActionPrompt& payload);
    void handle_end_game(const EndGame& payload);

    // Sets enqueue_count to the number of times this slot should be enqueued. Sets hibernate to
    // true if hibernating.
    void step(int& enqueue_count, bool& hibernate);

    bool game_started() const { return game_started_; }

   private:
    const Params& params() const { return shared_data_.params(); }
    bool step_chance();      // return true if terminal
    bool step_non_chance();  // return true if terminal
    void handle_terminal(const ValueTensor& outcome);
    void send_action_packet(const ActionResponse&);

    SharedData& shared_data_;
    const game_slot_index_t id_;
    player_array_t players_;
    HibernationNotifier hibernation_notifier_;

    // Initialized at the start of the game
    game_id_t game_id_;
    player_name_array_t player_names_;
    bool game_started_ = false;

    // Updated for each move
    StateHistory history_;
    ActionMask valid_actions_;
    bool play_noisily_;
    player_id_t prompted_player_id_;
    bool mid_yield_;

    // Used for synchronization in multithreaded case
    std::atomic<int> pending_drop_count_ = 0;
  };

  class SharedData {
   public:
    SharedData(const Params& params, int num_game_threads);
    ~SharedData();
    void register_player(seat_index_t seat, PlayerGenerator* gen);
    void init_socket();
    io::Socket* socket() const { return socket_; }
    PlayerGenerator* get_gen(player_id_t p) const { return players_[p]; }
    void start_session();
    void end_session();
    void shutdown();
    void init_game_slots();
    void run_hibernation_manager();
    HibernationManager* hibernation_manager() { return &hibernation_manager_; }
    int num_slots() const { return game_slots_.size(); }
    int num_game_threads() const { return num_game_threads_; }
    bool running() const { return running_; }

    GameSlot* next();  // returns nullptr if ready to shut down
    void enqueue(GameSlot*, int count);

    void handle_start_game(const GeneralPacket& packet);
    void handle_state_change(const GeneralPacket& packet);
    void handle_action_prompt(const GeneralPacket& packet);
    void handle_end_game(const GeneralPacket& packet);

   private:
    seat_generator_vec_t seat_generators_;   // temp storage
    player_generator_array_t players_ = {};  // indexed by player_id_t
    Params params_;
    int num_game_threads_;
    io::Socket* socket_ = nullptr;

    std::condition_variable cv_;
    mutable std::mutex mutex_;
    bool running_ = true;

    // Below fields mirror their usage in GameServer. See GameServer::SharedData comments for
    // details.
    std::vector<GameSlot*> game_slots_;
    std::queue<GameSlot*> queue_;

    HibernationManager hibernation_manager_;
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

  GameServerProxy(const Params& params, int num_game_threads)
      : shared_data_(params, num_game_threads) {}

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

  void run();

 private:
  const Params& params() const { return shared_data_.params(); }

  void create_threads();
  void launch_threads();
  void run_event_loop();
  void shutdown_threads();
  void join_threads();

  SharedData shared_data_;
  std::vector<GameThread*> threads_;
};

}  // namespace core

#include <inline/core/GameServerProxy.inl>
