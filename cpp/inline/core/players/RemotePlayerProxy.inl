#include <core/players/RemotePlayerProxy.hpp>

#include <sys/socket.h>

#include <core/Packet.hpp>

namespace core {

template <GameStateConcept GameState>
typename RemotePlayerProxy<GameState>::PacketDispatcher::dispatcher_map_t
    RemotePlayerProxy<GameState>::PacketDispatcher::dispatcher_map_;

template <GameStateConcept GameState>
RemotePlayerProxy<GameState>::PacketDispatcher*
RemotePlayerProxy<GameState>::PacketDispatcher::create(io::Socket* socket) {
  auto it = dispatcher_map_.find(socket);
  if (it != dispatcher_map_.end()) {
    return it->second;
  }
  auto dispatcher = new PacketDispatcher(socket);
  dispatcher_map_[socket] = dispatcher;
  return dispatcher;
}

template <GameStateConcept GameState>
void RemotePlayerProxy<GameState>::PacketDispatcher::start_all(int num_game_threads) {
  for (auto it : dispatcher_map_) {
    io::Socket* socket = it.first;
    PacketDispatcher* dispatcher = it.second;

    Packet<GameThreadInitialization> packet;
    packet.payload().num_game_threads = num_game_threads;
    packet.send_to(socket);

    Packet<GameThreadInitializationResponse> response;
    if (!response.read_from(socket)) {
      throw util::Exception("Unexpected socket close");
    }

    dispatcher->start();
  }
}

template <GameStateConcept GameState>
void RemotePlayerProxy<GameState>::PacketDispatcher::teardown() {
  for (auto it : dispatcher_map_) {
    PacketDispatcher* dispatcher = it.second;
    dispatcher->socket_->shutdown();
  }
}

template <GameStateConcept GameState>
void RemotePlayerProxy<GameState>::PacketDispatcher::add_player(
    RemotePlayerProxy<GameState>* player) {
  game_thread_id_t game_thread_id = player->game_thread_id_;
  player_id_t player_id = player->player_id_;

  util::clean_assert(player_id >= 0 && (int)player_id < kNumPlayers, "Invalid player_id (%d)",
                     (int)player_id);
  auto& vec = player_vec_array_[player_id];

  util::clean_assert((int)game_thread_id == (int)vec.size(), "Unexpected game_thread_id (%d != %d)",
                     (int)game_thread_id, (int)vec.size());

  vec.push_back(player);
}

template <GameStateConcept GameState>
void RemotePlayerProxy<GameState>::PacketDispatcher::start() {
  thread_ = new std::thread([&] { loop(); });
}

template <GameStateConcept GameState>
void RemotePlayerProxy<GameState>::PacketDispatcher::loop() {
  while (true) {  // TODO: track num listeners and change this condition to break when num listeners
                  // == 0
    GeneralPacket packet;
    if (!packet.read_from(socket_)) {
      // socket was shutdown()
      break;
    }

    auto type = packet.header().type;
    switch (type) {
      case PacketHeader::kActionDecision:
        handle_action(packet);
        break;
      default:
        throw util::Exception("Unexpected packet type: %d", (int)type);
    }
  }
}

template <GameStateConcept GameState>
RemotePlayerProxy<GameState>::PacketDispatcher::PacketDispatcher(io::Socket* socket)
    : socket_(socket) {}

template <GameStateConcept GameState>
void RemotePlayerProxy<GameState>::PacketDispatcher::handle_action(const GeneralPacket& packet) {
  const ActionDecision& payload = packet.payload_as<ActionDecision>();

  game_thread_id_t game_thread_id = payload.game_thread_id;
  player_id_t player_id = payload.player_id;

  RemotePlayerProxy* player = player_vec_array_[player_id][game_thread_id];

  // TODO: detect invalid packet and engage in a retry-protocol with remote player
  serializer_.deserialize_action_response(payload.dynamic_size_section.buf,
                                          &player->action_response_);
  player->cv_.notify_one();
}

template <GameStateConcept GameState>
RemotePlayerProxy<GameState>::RemotePlayerProxy(io::Socket* socket, player_id_t player_id,
                                                game_thread_id_t game_thread_id)
    : socket_(socket), player_id_(player_id), game_thread_id_(game_thread_id) {
  GameStateTypes::nullify_action(action_response_.action);
  auto dispatcher = PacketDispatcher::create(socket);
  dispatcher->add_player(this);
}

template <GameStateConcept GameState>
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

template <GameStateConcept GameState>
void RemotePlayerProxy<GameState>::receive_state_change(seat_index_t seat, const GameState& state,
                                                        const Action& action) {
  Packet<StateChange> packet;
  packet.payload().game_thread_id = game_thread_id_;
  packet.payload().player_id = player_id_;
  auto& section = packet.payload().dynamic_size_section;
  int buf_size =
      serializer_.serialize_state_change(section.buf, sizeof(section.buf), state, seat, action);
  packet.set_dynamic_section_size(buf_size);
  packet.send_to(socket_);
}

template <GameStateConcept GameState>
typename RemotePlayerProxy<GameState>::ActionResponse
RemotePlayerProxy<GameState>::get_action_response(const GameState& state,
                                                  const ActionMask& valid_actions) {
  state_ = &state;

  GameStateTypes::nullify_action(action_response_.action);

  Packet<ActionPrompt> packet;
  packet.payload().game_thread_id = game_thread_id_;
  packet.payload().player_id = player_id_;
  auto& section = packet.payload().dynamic_size_section;
  int buf_size =
      serializer_.serialize_action_prompt(section.buf, sizeof(section.buf), valid_actions);
  packet.set_dynamic_section_size(buf_size);
  packet.send_to(socket_);

  std::unique_lock lock(mutex_);
  cv_.wait(lock, [&] { return !GameStateTypes::is_nullified(action_response_.action); });
  return action_response_;
}

template <GameStateConcept GameState>
void RemotePlayerProxy<GameState>::end_game(const GameState& state, const GameOutcome& outcome) {
  Packet<EndGame> packet;
  packet.payload().game_thread_id = game_thread_id_;
  packet.payload().player_id = player_id_;
  auto& section = packet.payload().dynamic_size_section;
  int buf_size = serializer_.serialize_game_end(section.buf, sizeof(section.buf), outcome);
  packet.set_dynamic_section_size(buf_size);
  packet.send_to(socket_);
}

}  // namespace core
