#include <core/GameServerProxy.hpp>

#include <core/Constants.hpp>
#include <core/Globals.hpp>
#include <core/Packet.hpp>
#include <util/Exception.hpp>
#include <util/KeyValueDumper.hpp>
#include <util/LoggingUtil.hpp>

#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace core {

template <concepts::Game Game>
auto GameServerProxy<Game>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Remote GameServer options");

  return desc
    .template add_option<"remote-server">(
      po::value<std::string>(&remote_server)->default_value(remote_server),
      "Remote server to connect players to")
    .template add_option<"remote-port">(
      po::value<int>(&remote_port),
      "Remote port to connect players to. If not specified, run server in-process");
}

template <concepts::Game Game>
GameServerProxy<Game>::GameSlot::GameSlot(SharedData& shared_data, game_slot_index_t id)
    : shared_data_(shared_data),
      id_(id),
      hibernation_notifier_(shared_data.hibernation_manager(), id) {
  for (player_id_t p = 0; p < (player_id_t)kNumPlayers; ++p) {
    players_[p] = nullptr;
    PlayerGenerator* gen = shared_data_.get_gen(p);
    if (gen) {
      players_[p] = gen->generate(id);
    }
  }
}

template <concepts::Game Game>
GameServerProxy<Game>::GameSlot::~GameSlot() {
  for (player_id_t p = 0; p < (player_id_t)kNumPlayers; ++p) {
    delete players_[p];
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::handle_start_game(const StartGame& payload) {
  game_id_ = payload.game_id;
  payload.parse_player_names(player_names_);

  game_started_ = true;
  history_.initialize(Rules{});
  yield_state_ = kContinue;

  Player* player = players_[payload.player_id];
  player->init_game(game_id_, player_names_, payload.seat_assignment, nullptr);
  player->start_game();
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::handle_state_change(const StateChange& payload) {
  const char* buf = payload.dynamic_size_section.buf;

  seat_index_t seat = Rules::get_current_player(history_.current());
  ActionResponse action_response = *reinterpret_cast<const ActionResponse*>(buf);
  action_t action = action_response.action;
  Rules::apply(history_, action);

  Player* player = players_[payload.player_id];
  player->receive_state_change(seat, history_.current(), action);
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::handle_action_prompt(const ActionPrompt& payload) {
  const char* buf = payload.dynamic_size_section.buf;
  valid_actions_ = *reinterpret_cast<const ActionMask*>(buf);
  play_noisily_ = payload.play_noisily;
  prompted_player_id_ = payload.player_id;

  shared_data_.enqueue(this);
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::handle_end_game(const EndGame& payload) {
  const char* buf = payload.dynamic_size_section.buf;

  game_started_ = false;

  ValueTensor outcome = *reinterpret_cast<const ValueTensor*>(buf);
  Player* player = players_[payload.player_id];
  player->end_game(history_.current(), outcome);
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::step() {
  Player* player = players_[prompted_player_id_];

  ActionRequest request(history_.current(), valid_actions_);
  request.play_noisily = play_noisily_;
  request.hibernation_notifier = &hibernation_notifier_;
  ActionResponse response = player->get_action_response(request);
  yield_state_ = response.yield_instruction;
  if (yield_state_ == kContinue) {
    send_action_packet(response);
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::send_action_packet(const ActionResponse& response) {
  Packet<ActionDecision> packet;
  ActionDecision& decision = packet.payload();
  auto& section = decision.dynamic_size_section;

  decision.game_slot_index = id_;
  decision.player_id = prompted_player_id_;
  packet.set_dynamic_section_size(sizeof(response));
  memcpy(section.buf, &response, sizeof(response));
  packet.send_to(shared_data_.socket());
}

template <concepts::Game Game>
GameServerProxy<Game>::SharedData::SharedData(const Params& params, int num_game_threads)
    : params_(params), num_game_threads_(num_game_threads) {
  util::clean_assert(params_.remote_port > 0, "Remote port must be specified");
  socket_ = io::Socket::create_client_socket(params_.remote_server, params_.remote_port);
  LOG_INFO("Connected to the server!");
}

template <concepts::Game Game>
GameServerProxy<Game>::SharedData::~SharedData() {
  for (auto gen : players_) {
    delete gen;
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::register_player(seat_index_t seat, PlayerGenerator* gen) {
  // TODO: assert that we are not constructing MCTS-T players, since the MCTS-T implementation
  // implicitly assumes that all MCTS-T agents are running in the same process.
  std::string name = gen->get_name();
  util::clean_assert(name.size() + 1 < kMaxNameLength, "Player name too long (\"%s\" size=%d)",
                     name.c_str(), (int)name.size());
  seat_generators_.emplace_back(seat, gen);
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::init_socket() {
  int n = seat_generators_.size();
  for (int i = 0; i < n; ++i) {
    SeatGenerator& seat_generator = seat_generators_[i];
    seat_index_t seat = seat_generator.seat;
    PlayerGenerator* gen = seat_generator.gen;

    std::string registered_name = gen->get_name();
    if (registered_name.empty()) {
      registered_name = gen->get_default_name();
    }

    LOG_INFO("Registering player \"{}\" at seat {}", registered_name, seat);

    Packet<Registration> send_packet;
    Registration& registration = send_packet.payload();
    registration.remaining_requests = n - i - 1;
    registration.max_simultaneous_games = gen->max_simultaneous_games();
    registration.requested_seat = seat;
    send_packet.set_player_name(registered_name);
    send_packet.send_to(socket_);

    Packet<RegistrationResponse> recv_packet;
    if (!recv_packet.read_from(socket_)) {
      throw util::Exception("Unexpected socket close");
    }
    const RegistrationResponse& response = recv_packet.payload();
    player_id_t player_id = response.player_id;
    std::string name = response.dynamic_size_section.player_name;

    util::clean_assert(player_id >= 0 && player_id < kNumPlayers, "Invalid player_id: %d",
                       (int)player_id);
    util::clean_assert(registered_name.empty() || registered_name == name,
                       "Unexpected name in response: \"%s\" != \"%s\"", registered_name.c_str(),
                       name.c_str());

    gen->set_name(name);
    players_[player_id] = gen;
    LOG_INFO("Registered player \"{}\" at seat {} (player_id:{})", name, seat, player_id);
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::start_session() {
  for (auto& sg : seat_generators_) {
    sg.gen->start_session();
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::end_session() {
  for (auto& sg : seat_generators_) {
    sg.gen->end_session();
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::shutdown() {
  std::unique_lock lock(mutex_);
  running_ = false;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::init_game_slots() {
  Packet<GameThreadInitialization> recv_packet;
  if (!recv_packet.read_from(socket_)) {
    throw util::Exception("Unexpected socket close");
  }
  int num_game_slots = recv_packet.payload().num_game_slots;
  game_slots_.reserve(num_game_slots);
  for (int i = 0; i < num_game_slots; ++i) {
    game_slots_.push_back(new GameSlot(*this, i));
  }
  num_game_threads_ = std::min(num_game_threads_, num_game_slots);
  core::Globals::num_game_threads = num_game_threads_;

  Packet<GameThreadInitializationResponse> send_packet;
  send_packet.send_to(socket_);
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::run_hibernation_manager() {
  hibernation_manager_.run([this](game_slot_index_t slot_id) {
    std::unique_lock lock(mutex_);
    GameSlot* slot = game_slots_[slot_id];
    queue_.push(slot);
    lock.unlock();

    cv_.notify_all();
  });
}

template <concepts::Game Game>
typename GameServerProxy<Game>::GameSlot* GameServerProxy<Game>::SharedData::next() {
  std::unique_lock lock(mutex_);

  if (queue_.empty()) {
    cv_.wait(lock, [&] { return !running_ || !queue_.empty(); });
  }

  if (!running_) {
    return nullptr;
  }

  GameSlot* slot = queue_.front();
  queue_.pop();
  return slot;
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::enqueue(GameSlot* slot) {
  std::unique_lock lock(mutex_);
  bool empty = queue_.empty();
  queue_.push(slot);
  if (empty) {
    lock.unlock();
    cv_.notify_all();
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::handle_start_game(const GeneralPacket& packet) {
  const StartGame& payload = packet.payload_as<StartGame>();
  game_slots_[payload.game_slot_index]->handle_start_game(payload);
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::handle_state_change(const GeneralPacket& packet) {
  const StateChange& payload = packet.payload_as<StateChange>();
  game_slots_[payload.game_slot_index]->handle_state_change(payload);
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::handle_action_prompt(const GeneralPacket& packet) {
  const ActionPrompt& payload = packet.payload_as<ActionPrompt>();
  game_slots_[payload.game_slot_index]->handle_action_prompt(payload);
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::handle_end_game(const GeneralPacket& packet) {
  const EndGame& payload = packet.payload_as<EndGame>();
  game_slots_[payload.game_slot_index]->handle_end_game(payload);
}

template <concepts::Game Game>
GameServerProxy<Game>::GameThread::GameThread(SharedData& shared_data, game_thread_id_t id)
    : shared_data_(shared_data), id_(id) {
}

template <concepts::Game Game>
GameServerProxy<Game>::GameThread::~GameThread() {
  join();
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameThread::join() {
  if (thread_.joinable()) thread_.join();
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameThread::launch() {
  thread_ = std::thread([&] { run(); });
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameThread::run() {
  while (shared_data_.running()) {
    GameSlot* slot = shared_data_.next();
    if (!slot) return;

    util::release_assert(slot->game_started());
    slot->step();

    if (slot->game_started()) {
      if (slot->yield_state() == kYield) {
        shared_data_.enqueue(slot);
      }
    }
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::run() {
  Game::static_init();
  shared_data_.init_socket();
  shared_data_.init_game_slots();
  shared_data_.start_session();
  shared_data_.run_hibernation_manager();
  create_threads();
  launch_threads();
  run_event_loop();
  shared_data_.shutdown();
  join_threads();
  shared_data_.end_session();
  util::KeyValueDumper::flush();
}

template <concepts::Game Game>
void GameServerProxy<Game>::create_threads() {
  int num_threads = std::min(shared_data_.num_slots(), shared_data_.num_game_threads());
  for (int t = 0; t < num_threads; ++t) {
    GameThread* thread = new GameThread(shared_data_, t);
    threads_.push_back(thread);
  }
  core::Globals::num_game_threads = num_threads;
}

template <concepts::Game Game>
void GameServerProxy<Game>::launch_threads() {
  for (auto thread : threads_) {
    thread->launch();
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::run_event_loop() {
  while (true) {
    GeneralPacket response_packet;
    if (!response_packet.read_from(shared_data_.socket())) {
      break;
    }

    auto type = response_packet.header().type;
    switch (type) {
      case PacketHeader::kStartGame:
        shared_data_.handle_start_game(response_packet);
        break;
      case PacketHeader::kStateChange:
        shared_data_.handle_state_change(response_packet);
        break;
      case PacketHeader::kActionPrompt:
        shared_data_.handle_action_prompt(response_packet);
        break;
      case PacketHeader::kEndGame:
        shared_data_.handle_end_game(response_packet);
        break;
      default:
        throw util::Exception("Unexpected packet type: {}", (int)type);
    }
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::shutdown_threads() {
  for (auto thread : threads_) {
    thread->shutdown();
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::join_threads() {
  for (auto thread : threads_) {
    thread->join();
  }
}

}  // namespace core
