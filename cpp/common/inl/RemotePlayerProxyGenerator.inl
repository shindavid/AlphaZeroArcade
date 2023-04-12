#include <common/RemotePlayerProxyGenerator.hpp>

#include <common/Packet.hpp>
#include <util/Exception.hpp>

#include <unistd.h>

namespace common {

template <GameStateConcept GameState>
void RemotePlayerProxyGenerator<GameState>::initialize(
    const std::string& name, int socket_descriptor, player_id_t player_id)
{
  this->set_name(name);
  socket_descriptor_ = socket_descriptor;
  player_id_ = player_id;
}

template <GameStateConcept GameState>
AbstractPlayer<GameState>* RemotePlayerProxyGenerator<GameState>::generate(game_thread_id_t game_thread_id) {
  util::clean_assert(initialized(), "RemotePlayerProxyGenerator::generate() called before initialized");

  Packet<GameThreadInitialization> send_packet;
  send_packet.payload().game_thread_id = game_thread_id;
  send_packet.send_to(socket_descriptor_);

  Packet<GameThreadInitializationResponse> recv_packet;
  recv_packet.read_from(socket_descriptor_);
  int max_simultaneous_games = recv_packet.payload().max_simultaneous_games;

  return new RemotePlayerProxy<GameState>(socket_descriptor_, player_id_, game_thread_id, max_simultaneous_games);
}

template <GameStateConcept GameState>
void RemotePlayerProxyGenerator<GameState>::end_session() {
  close(socket_descriptor_);
}

}  // namespace common
