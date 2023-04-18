#include <common/RemotePlayerProxy.hpp>

#include <sys/socket.h>

#include <common/Packet.hpp>

namespace common {

template<GameStateConcept GameState>
typename RemotePlayerProxy<GameState>::PacketDispatcher::dispatcher_map_t
RemotePlayerProxy<GameState>::PacketDispatcher::dispatcher_map_;

template<GameStateConcept GameState>
RemotePlayerProxy<GameState>::PacketDispatcher* RemotePlayerProxy<GameState>::PacketDispatcher::create(
    io::Socket* socket)
{
  auto it = dispatcher_map_.find(socket);
  if (it != dispatcher_map_.end()) {
    return it->second;
  }
  auto dispatcher = new PacketDispatcher(socket);
  dispatcher_map_[socket] = dispatcher;
  return dispatcher;
}

template<GameStateConcept GameState>
void RemotePlayerProxy<GameState>::PacketDispatcher::start_all(int num_game_threads) {
  for (auto it : dispatcher_map_) {
    io::Socket* socket = it.first;
    PacketDispatcher* dispatcher = it.second;

    Packet<GameThreadInitialization> packet;
    packet.payload().num_game_threads = num_game_threads;
    packet.send_to(socket);

    Packet<GameThreadInitializationResponse> response;
    response.read_from(socket);

    dispatcher->start();
  }
}

template<GameStateConcept GameState>
void RemotePlayerProxy<GameState>::PacketDispatcher::add_player(RemotePlayerProxy<GameState> *player) {
  game_thread_id_t game_thread_id = player->game_thread_id_;
  player_id_t player_id = player->player_id_;

  util::clean_assert(player_id >= 0 && (int)player_id < kNumPlayers, "Invalid player_id (%d)", (int)player_id);
  auto& vec = player_vec_array_[player_id];

  util::clean_assert((int)game_thread_id == (int)vec.size(), "Unexpected game_thread_id (%d != %d)",
                     (int)game_thread_id, (int)vec.size());

  vec.push_back(player);
}

template<GameStateConcept GameState>
void RemotePlayerProxy<GameState>::PacketDispatcher::start() {
  thread_ = new std::thread([&] { loop(); });
}

template<GameStateConcept GameState>
void RemotePlayerProxy<GameState>::PacketDispatcher::loop() {
  while (true) {  // TODO: track num listeners and change this condition to break when num listeners == 0
    GeneralPacket packet;
    packet.read_from(socket_);

    auto type = packet.header().type;
    switch (type) {
      case PacketHeader::kAction:
        handle_action(packet);
        break;
      default:
        throw util::Exception("Unexpected packet type: %d", (int) type);
    }
  }
}

template<GameStateConcept GameState>
RemotePlayerProxy<GameState>::PacketDispatcher::PacketDispatcher(io::Socket* socket) : socket_(socket) {}

template<GameStateConcept GameState>
void RemotePlayerProxy<GameState>::PacketDispatcher::handle_action(const GeneralPacket& packet) {
  const Action& payload = packet.payload_as<Action>();

  game_thread_id_t game_thread_id = payload.game_thread_id;
  player_id_t player_id = payload.player_id;

  RemotePlayerProxy* player = player_vec_array_[player_id][game_thread_id];

  // TODO: detect invalid packet and engage in a retry-protocol with remote player
  player->state_->deserialize_action(payload.dynamic_size_section.buf, &player->action_);
  player->cv_.notify_one();
}

template<GameStateConcept GameState>
RemotePlayerProxy<GameState>::RemotePlayerProxy(
    io::Socket* socket, player_id_t player_id, game_thread_id_t game_thread_id)
: socket_(socket)
, player_id_(player_id)
, game_thread_id_(game_thread_id)
{
  auto dispatcher = PacketDispatcher::create(socket);
  dispatcher->add_player(this);
}

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
  state_ = &state;
  action_ = -1;

  Packet<ActionPrompt> packet;
  packet.payload().game_thread_id = game_thread_id_;
  packet.payload().player_id = player_id_;
  auto buf = packet.payload().dynamic_size_section.buf;
  int buf_size = state.serialize_action_prompt(buf, sizeof(buf), valid_actions);
  packet.set_dynamic_section_size(buf_size);
  packet.send_to(socket_);

  std::unique_lock lock(mutex_);
  cv_.wait(lock, [&] { return action_ != -1; });
  return action_;
}

template<GameStateConcept GameState>
void RemotePlayerProxy<GameState>::end_game(const GameState& state, const GameOutcome& outcome) {
  Packet<EndGame> packet;
  packet.payload().game_thread_id = game_thread_id_;
  packet.payload().player_id = player_id_;
  auto buf = packet.payload().dynamic_size_section.buf;
  packet.set_dynamic_section_size(state.serialize_game_end(buf, sizeof(buf), outcome));
  packet.send_to(socket_);
}

}  // namespace common
