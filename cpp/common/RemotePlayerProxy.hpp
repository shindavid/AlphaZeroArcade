#pragma once

#include <condition_variable>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/Packet.hpp>
#include <util/SocketUtil.hpp>

namespace common {

/*
 * In a server-client setup, the server process will create a RemotePlayerProxy to act as a proxy for remote
 * players. The RemotePlayerProxy will communicate with the remote player over a socket.
 */
template<GameStateConcept GameState>
class RemotePlayerProxy : public AbstractPlayer<GameState> {
public:
  static constexpr int kNumPlayers = GameState::kNumPlayers;
  using GameStateTypes = common::GameStateTypes<GameState>;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using Player = AbstractPlayer<GameState>;
  using player_vec_t = std::vector<RemotePlayerProxy*>;  // keyed by game_thread_id_t
  using player_vec_array_t = std::array<player_vec_t, kNumPlayers>;

  class PacketDispatcher {
  public:
    static PacketDispatcher* create(io::Socket* socket);
    static void start_all(int num_game_threads);
    static void teardown();

    void add_player(RemotePlayerProxy* player);
    void start();

  private:
    PacketDispatcher(io::Socket* socket);
    PacketDispatcher(const PacketDispatcher&) = delete;
    PacketDispatcher& operator=(const PacketDispatcher&) = delete;

    void loop();
    void handle_action(const GeneralPacket& packet);

    using dispatcher_map_t = std::map<io::Socket*, PacketDispatcher*>;

    static dispatcher_map_t dispatcher_map_;

    std::thread* thread_ = nullptr;
    io::Socket* socket_;
    player_vec_array_t player_vec_array_;
  };

  RemotePlayerProxy(io::Socket* socket, player_id_t player_id, game_thread_id_t game_thread_id);

  void start_game() override;
  void receive_state_change(seat_index_t, const GameState&, action_index_t) override;
  action_index_t get_action(const GameState&, const ActionMask&) override;
  void end_game(const GameState&, const GameOutcome&) override;

private:
  std::condition_variable cv_;
  mutable std::mutex mutex_;

  const GameState* state_ = nullptr;
  action_index_t action_ = -1;

  io::Socket* socket_;
  const player_id_t player_id_;
  const game_thread_id_t game_thread_id_;
};

}  // namespace common

#include <common/inl/RemotePlayerProxy.inl>
