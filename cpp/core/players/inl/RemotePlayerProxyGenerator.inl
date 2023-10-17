#include <core/players/RemotePlayerProxyGenerator.hpp>

#include <core/Packet.hpp>
#include <util/Exception.hpp>

#include <unistd.h>

namespace core {

template <GameStateConcept GameState>
void RemotePlayerProxyGenerator<GameState>::initialize(io::Socket* socket,
                                                       int max_simultaneous_games,
                                                       player_id_t player_id) {
  socket_ = socket;
  max_simultaneous_games_ = max_simultaneous_games;
  player_id_ = player_id;
}

template <GameStateConcept GameState>
AbstractPlayer<GameState>* RemotePlayerProxyGenerator<GameState>::generate(
    game_thread_id_t game_thread_id) {
  util::clean_assert(initialized(),
                     "RemotePlayerProxyGenerator::generate() called before initialized");
  return new RemotePlayerProxy<GameState>(socket_, player_id_, game_thread_id);
}

template <GameStateConcept GameState>
void RemotePlayerProxyGenerator<GameState>::end_session() {
  RemotePlayerProxy<GameState>::PacketDispatcher::teardown();
}

template <GameStateConcept GameState>
int RemotePlayerProxyGenerator<GameState>::max_simultaneous_games() const {
  util::clean_assert(max_simultaneous_games_ >= 0,
                     "RemotePlayerProxyGenerator::%s() called before initialized", __func__);
  return max_simultaneous_games_;
}

}  // namespace core
