#include <common/RemotePlayerProxy.hpp>

#include <sys/socket.h>

#include <common/Packet.hpp>

namespace common {

template<GameStateConcept GameState>
RemotePlayerProxy<GameState>::RemotePlayerProxy(
    io::Socket* socket, player_id_t player_id, game_thread_id_t game_thread_id, int max_simultaneous_games)
: socket_(socket)
, player_id_(player_id)
, game_thread_id_(game_thread_id)
, max_simultaneous_games_(max_simultaneous_games) {}

template<GameStateConcept GameState>
void RemotePlayerProxy<GameState>::start_game() {
  Packet<StartGame> packet;
  StartGame& payload = packet.payload();
  payload.game_id = this->get_game_id();
  payload.player_id = player_id_;
  payload.game_thread_id = game_thread_id_;
  payload.seat_assignment = this->get_my_seat();
  payload.load_player_names(packet, this->get_player_names());
  packet.send_to(socket_);
}

template<GameStateConcept GameState>
void RemotePlayerProxy<GameState>::receive_state_change(
    seat_index_t seat, const GameState& state, action_index_t action)
{
  Packet<StateChange> packet;
  packet.payload().game_thread_id = game_thread_id_;
  packet.payload().player_id = player_id_;
  auto buf = packet.payload().dynamic_size_section.buf;
  packet.set_dynamic_section_size(state.serialize_state_change(buf, sizeof(buf), seat, action));
  packet.send_to(socket_);
}

template<GameStateConcept GameState>
action_index_t RemotePlayerProxy<GameState>::get_action(const GameState& state, const ActionMask& valid_actions) {
  Packet<ActionPrompt> packet;
  auto buf = packet.payload().dynamic_size_section.buf;
  int buf_size = state.serialize_action_prompt(buf, sizeof(buf), valid_actions);
  packet.set_dynamic_section_size(buf_size);
  packet.send_to(socket_);

  // TODO: detect invalid packet and engage in a retry-protocol with remote player
  Packet<Action> response;
  response.read_from(socket_);
  action_index_t action;
  state.deserialize_action(response.payload().dynamic_size_section.buf, &action);
  return action;
}

template<GameStateConcept GameState>
void RemotePlayerProxy<GameState>::end_game(const GameState&, const GameOutcome&) {
  util::clean_assert(false, "Not implemented yet.");
}

}  // namespace common
