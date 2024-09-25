#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

#include <core/AbstractPlayerGenerator.hpp>
#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/Packet.hpp>
#include <util/CppUtil.hpp>
#include <util/SocketUtil.hpp>

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
  using ValueArray = Game::Types::ValueArray;
  using PlayerGenerator = AbstractPlayerGenerator<Game>;
  using player_generator_array_t = std::array<PlayerGenerator*, kNumPlayers>;
  using Player = AbstractPlayer<Game>;
  using player_name_array_t = Player::player_name_array_t;
  using player_array_t = std::array<Player*, kNumPlayers>;
  using player_vec_t = std::vector<Player*>;

  struct seat_generator_t {
    seat_index_t seat;
    PlayerGenerator* gen;
  };
  using seat_generator_vec_t = std::vector<seat_generator_t>;

  struct Params {
    auto make_options_description();

    std::string remote_server = "localhost";
    int remote_port = 0;
  };

  class SharedData {
   public:
    SharedData(const Params& params);
    ~SharedData();
    void register_player(seat_index_t seat, PlayerGenerator* gen);
    void init_socket();
    io::Socket* socket() const { return socket_; }
    PlayerGenerator* get_gen(player_id_t p) const { return players_[p]; }
    void end_session();

   private:
    seat_generator_vec_t seat_generators_;   // temp storage
    player_generator_array_t players_ = {};  // indexed by player_id_t
    Params params_;
    io::Socket* socket_ = nullptr;
  };

  class PlayerThread {
   public:
    PlayerThread(SharedData& shared_data, Player* player, game_thread_id_t game_thread_id,
                 player_id_t player_id);
    ~PlayerThread();

    void handle_start_game(const StartGame& payload);
    void handle_state_change(const StateChange& payload);
    void handle_action_prompt(const ActionPrompt& payload);
    void handle_end_game(const EndGame& payload);

    void join() {
      if (thread_ && thread_->joinable()) thread_->join();
    }
    void stop();

   private:
    void send_action_packet(const ActionResponse&);
    void run();

    std::condition_variable cv_;
    mutable std::mutex mutex_;

    SharedData& shared_data_;
    Player* const player_;
    const game_thread_id_t game_thread_id_;
    const player_id_t player_id_;
    std::thread* thread_ = nullptr;

    StateHistory history_;

    // below fields are used for inter-thread communication
    ActionMask valid_actions_;
    bool active_ = true;
    bool ready_to_get_action_ = false;
  };
  using thread_array_t = std::array<PlayerThread*, kNumPlayers>;  // indexed by player_id_t
  using thread_vec_t = std::vector<thread_array_t>;               // index by game_thread_id_t

  GameServerProxy(const Params& params) : shared_data_(params) {}
  ~GameServerProxy();

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
  void init_player_threads();
  void destroy_player_threads();
  void handle_start_game(const GeneralPacket& packet);
  void handle_state_change(const GeneralPacket& packet);
  void handle_action_prompt(const GeneralPacket& packet);
  void handle_end_game(const GeneralPacket& packet);

  SharedData shared_data_;
  thread_vec_t thread_vec_;
};

}  // namespace core

#include <inline/core/GameServerProxy.inl>
