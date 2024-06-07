#pragma once

#include <condition_variable>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/Packet.hpp>
#include <util/SocketUtil.hpp>

namespace core {

/*
 * In a server-client setup, the server process will create a RemotePlayerProxy to act as a proxy
 * for remote players. The RemotePlayerProxy will communicate with the remote player over a socket.
 */
template <concepts::Game Game>
class RemotePlayerProxy : public AbstractPlayer<Game> {
 public:
  static constexpr int kNumPlayers = Game::kNumPlayers;
  using FullState = typename Game::FullState;
  using ActionMask = typename Game::ActionMask;
  using ValueArray = typename Game::ValueArray;
  using Player = AbstractPlayer<Game>;
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
  void receive_state_change(seat_index_t, const FullState&, action_t) override;
  ActionResponse get_action_response(const FullState&, const ActionMask&) override;
  void end_game(const FullState&, const ValueArray&) override;

 private:
  std::condition_variable cv_;
  mutable std::mutex mutex_;

  const FullState* state_ = nullptr;
  ActionResponse action_response_;

  io::Socket* socket_;
  const player_id_t player_id_;
  const game_thread_id_t game_thread_id_;
};

}  // namespace core

#include <inline/core/players/RemotePlayerProxy.inl>
