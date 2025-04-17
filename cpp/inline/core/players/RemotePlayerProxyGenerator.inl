#include <core/players/RemotePlayerProxyGenerator.hpp>

#include <core/Packet.hpp>
#include <util/Exception.hpp>

#include <unistd.h>

namespace core {

template <concepts::Game Game>
void RemotePlayerProxyGenerator<Game>::initialize(io::Socket* socket, int max_simultaneous_games,
                                                  player_id_t player_id) {
  socket_ = socket;
  max_simultaneous_games_ = max_simultaneous_games;
  player_id_ = player_id;
}

template <concepts::Game Game>
AbstractPlayer<Game>* RemotePlayerProxyGenerator<Game>::generate(game_thread_id_t game_thread_id) {
  util::clean_assert(initialized(),
                     "RemotePlayerProxyGenerator::generate() called before initialized");
  return new RemotePlayerProxy<Game>(socket_, player_id_, game_thread_id);
}

template <concepts::Game Game>
void RemotePlayerProxyGenerator<Game>::end_session() {
  RemotePlayerProxy<Game>::PacketDispatcher::teardown();
}

template <concepts::Game Game>
int RemotePlayerProxyGenerator<Game>::max_simultaneous_games() const {
  util::clean_assert(max_simultaneous_games_ >= 0,
                     "RemotePlayerProxyGenerator::%s() called before initialized", __func__);
  return max_simultaneous_games_;
}

}  // namespace core
