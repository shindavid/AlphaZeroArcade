#pragma once

#include <common/RemotePlayerProxy.hpp>

#include <common/Packet.hpp>

namespace common {

template<GameStateConcept GameState>
RemotePlayerProxy<GameState>::RemotePlayerProxy(const std::string& name, int socket_descriptor)
    : Player(util::create_string("%s@%d", name.c_str(), socket_descriptor))
      , socket_descriptor_(socket_descriptor) {}

template<GameStateConcept GameState>
void RemotePlayerProxy<GameState>::start_game(
    game_id_t game_id, const player_array_t& players, player_index_t seat_assignment)
{
  StartGamePayload payload{game_id, players.size(), seat_assignment};
  Packet::to_socket(socket_descriptor_, PacketHeader::kStartGame, payload);
}

template<GameStateConcept GameState>
void RemotePlayerProxy<GameState>::receive_state_change(
    player_index_t player, const GameState& state, action_index_t action, const GameOutcome& outcome)
{
  char buf[1024];
  int buf_size = state.serialize_state_change(buf, sizeof(buf), player, action, outcome);
  Packet::to_socket(socket_descriptor_, PacketHeader::kStateChange, buf, buf_size);
}

template<GameStateConcept GameState>
action_index_t RemotePlayerProxy<GameState>::get_action(const GameState& state, const ActionMask& valid_actions) {
  char buf[1024];

  int buf_size = state.serialize_action_prompt(buf, sizeof(buf), valid_actions);
  Packet::to_socket(socket_descriptor_, PacketHeader::kActionPrompt, buf, buf_size);

  Packet response = Packet::from_socket(socket_descriptor_, buf, sizeof(buf));
  if (response.header.type != PacketHeader::kAction) {
    throw util::Exception("Expected kAction, got %d", response.header.type);
  }
  action_index_t action;
  state.deserialize_action(response.payload, &action);
  return action;
}

}  // namespace common
