#include <core/players/RemotePlayerProxy.hpp>

#include <core/BasicTypes.hpp>
#include <core/Packet.hpp>

#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>

#include <sys/socket.h>

namespace core {

template <concepts::Game Game>
typename RemotePlayerProxy<Game>::PacketDispatcher::dispatcher_map_t
    RemotePlayerProxy<Game>::PacketDispatcher::dispatcher_map_;

template <concepts::Game Game>
RemotePlayerProxy<Game>::PacketDispatcher*
RemotePlayerProxy<Game>::PacketDispatcher::create(io::Socket* socket) {
  auto it = dispatcher_map_.find(socket);
  if (it != dispatcher_map_.end()) {
    return it->second;
  }
  auto dispatcher = new PacketDispatcher(socket);
  dispatcher_map_[socket] = dispatcher;
  return dispatcher;
}

template <concepts::Game Game>
void RemotePlayerProxy<Game>::PacketDispatcher::start_all(int num_game_slots) {
  for (auto it : dispatcher_map_) {
    io::Socket* socket = it.first;
    PacketDispatcher* dispatcher = it.second;

    Packet<GameThreadInitialization> packet;
    packet.payload().num_game_slots = num_game_slots;
    packet.send_to(socket);

    Packet<GameThreadInitializationResponse> response;
    if (!response.read_from(socket)) {
      throw util::Exception("Unexpected socket close");
    }

    dispatcher->start();
  }
}

template <concepts::Game Game>
void RemotePlayerProxy<Game>::PacketDispatcher::teardown() {
  for (auto it : dispatcher_map_) {
    PacketDispatcher* dispatcher = it.second;
    dispatcher->socket_->shutdown();
  }
}

template <concepts::Game Game>
void RemotePlayerProxy<Game>::PacketDispatcher::add_player(
    RemotePlayerProxy<Game>* player) {
  game_slot_index_t game_slot_index = player->game_slot_index_;
  player_id_t player_id = player->player_id_;

  CLEAN_ASSERT(player_id >= 0 && (int)player_id < kNumPlayers, "Invalid player_id ({})",
                     player_id);
  auto& vec = player_vec_array_[player_id];

  CLEAN_ASSERT((int)game_slot_index == (int)vec.size(),
                     "Unexpected game_slot_index ({} != {})", game_slot_index, vec.size());

  vec.push_back(player);
}

template <concepts::Game Game>
void RemotePlayerProxy<Game>::PacketDispatcher::start() {
  thread_ = new mit::thread([&] { loop(); });
}

template <concepts::Game Game>
void RemotePlayerProxy<Game>::PacketDispatcher::loop() {
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
        throw util::Exception("Unexpected packet type: {}", type);
    }
  }
}

template <concepts::Game Game>
RemotePlayerProxy<Game>::PacketDispatcher::PacketDispatcher(io::Socket* socket)
    : socket_(socket) {}

template <concepts::Game Game>
void RemotePlayerProxy<Game>::PacketDispatcher::handle_action(const GeneralPacket& packet) {
  const ActionDecision& payload = packet.payload_as<ActionDecision>();

  game_slot_index_t game_slot_index = payload.game_slot_index;
  player_id_t player_id = payload.player_id;

  RemotePlayerProxy* player = player_vec_array_[player_id][game_slot_index];

  // TODO: detect invalid packet and engage in a retry-protocol with remote player
  const char* buf = payload.dynamic_size_section.buf;
  std::memcpy(&player->action_response_, buf, sizeof(player->action_response_));
  const YieldNotificationUnit& unit = player->yield_notification_unit_;
  unit.yield_manager->notify(unit);
}

template <concepts::Game Game>
RemotePlayerProxy<Game>::RemotePlayerProxy(io::Socket* socket, player_id_t player_id,
                                           game_slot_index_t game_slot_index)
    : socket_(socket), player_id_(player_id), game_slot_index_(game_slot_index) {
  action_response_.action = -1;
  auto dispatcher = PacketDispatcher::create(socket);
  dispatcher->add_player(this);
}

template <concepts::Game Game>
void RemotePlayerProxy<Game>::start_game() {
  Packet<StartGame> packet;
  StartGame& payload = packet.payload();
  payload.game_id = this->get_game_id();
  payload.player_id = player_id_;
  payload.game_slot_index = game_slot_index_;
  payload.seat_assignment = this->get_my_seat();
  payload.load_player_names(packet, this->get_player_names());
  packet.send_to(socket_);
}

template <concepts::Game Game>
void RemotePlayerProxy<Game>::receive_state_change(seat_index_t seat, const State& state,
                                                   action_t action) {
  ActionResponse action_response(action);
  Packet<StateChange> packet;
  packet.payload().game_slot_index = game_slot_index_;
  packet.payload().player_id = player_id_;
  auto& section = packet.payload().dynamic_size_section;
  memcpy(section.buf, &action_response, sizeof(action_response));
  packet.set_dynamic_section_size(sizeof(action_response));
  packet.send_to(socket_);
}

template <concepts::Game Game>
typename RemotePlayerProxy<Game>::ActionResponse RemotePlayerProxy<Game>::get_action_response(
    const ActionRequest& request) {
  if (yielding_) {
    yielding_ = false;
    return action_response_;
  }
  const ActionMask& valid_actions = request.valid_actions;

  action_response_.action = -1;

  RELEASE_ASSERT(request.notification_unit.context_id == 0,
                       "Unexpected context_id: {}", request.notification_unit.context_id);

  Packet<ActionPrompt> packet;
  packet.payload().game_slot_index = game_slot_index_;
  packet.payload().player_id = player_id_;
  packet.payload().play_noisily = request.play_noisily;
  auto& section = packet.payload().dynamic_size_section;
  memcpy(section.buf, &valid_actions, sizeof(valid_actions));
  packet.set_dynamic_section_size(sizeof(valid_actions));
  packet.send_to(socket_);

  yield_notification_unit_ = request.notification_unit;
  yielding_ = true;
  return ActionResponse::yield();
}

template <concepts::Game Game>
void RemotePlayerProxy<Game>::end_game(const State& state, const ValueTensor& outcome) {
  Packet<EndGame> packet;
  packet.payload().game_slot_index = game_slot_index_;
  packet.payload().player_id = player_id_;
  auto& section = packet.payload().dynamic_size_section;
  memcpy(section.buf, &outcome, sizeof(outcome));
  packet.set_dynamic_section_size(sizeof(outcome));
  packet.send_to(socket_);
}

}  // namespace core
