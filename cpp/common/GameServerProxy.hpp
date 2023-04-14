#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

#include <common/AbstractPlayerGenerator.hpp>
#include <common/BasicTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/Packet.hpp>

namespace common {

template <GameStateConcept GameState>
class GameServerProxy {
public:
  static constexpr int kNumPlayers = GameState::kNumPlayers;

  using PlayerGenerator = AbstractPlayerGenerator<GameState>;
  using player_generator_array_t = std::array<PlayerGenerator*, kNumPlayers>;
  using Player = AbstractPlayer<GameState>;
  using player_name_array_t = typename Player::player_name_array_t;
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

  struct player_state_t {
    GameState* state = nullptr;
    Player* player = nullptr;
  };
  using player_state_array_t = std::array<player_state_t, kNumPlayers>;

  class SharedData {
  public:
    SharedData(const Params& params);
    void register_player(seat_index_t seat, PlayerGenerator* gen);
    void init_socket();
    int socket_desc() const { return socket_desc_; }
    PlayerGenerator* get_gen(player_id_t p) const { return player_generators_[p]; }

  private:
    seat_generator_vec_t seat_generators_;  // temp storage
    player_generator_array_t player_generators_ = {};  // indexed by player_id_t
    Params params_;
    int socket_desc_ = -1;
  };

  class GameThread {
  public:
    GameThread(SharedData& shared_data, game_thread_id_t id);
    ~GameThread();

    void handle_start_game(const StartGame& payload);
    void handle_state_change(const StateChange& payload);

    void join() { if (thread_ && thread_->joinable()) thread_->join(); }
    void launch();
    int max_simultaneous_games() const { return max_simultaneous_games_; }

  private:
    void run();

    std::condition_variable cv_;
    mutable std::mutex mutex_;
    SharedData& shared_data_;
    player_state_array_t player_states_;
    game_thread_id_t id_;
    std::thread* thread_ = nullptr;
    int max_simultaneous_games_ = 0;
  };
  using thread_vec_t = std::vector<GameThread*>;  // index by game_thread_id_t

  GameServerProxy(const Params& params) : shared_data_(params) {}

  /*
   * If seat is not specified, then the player generator is assigned a random seat.
   *
   * Otherwise, the player generated is assigned the specified seat.
   *
   * The player generator is assigned a unique player_id_t (0, 1, 2, ...), according to the order in which the
   * registrations are made. This value is returned by this function. When aggregate game outcome stats are reported,
   * they are aggregated by player_id_t.
   */
  void register_player(PlayerGenerator* gen) { register_player(-1, gen); }
  void register_player(seat_index_t seat, PlayerGenerator* gen) { shared_data_.register_player(seat, gen); }

  void run();

private:
  void handle_game_thread_initialization(const GeneralPacket& packet);
  void handle_start_game(const GeneralPacket& packet);
  void handle_state_change(const GeneralPacket& packet);

  SharedData shared_data_;
  thread_vec_t thread_vec_;
};

}  // namespace common

#include <common/inl/GameServerProxy.inl>
