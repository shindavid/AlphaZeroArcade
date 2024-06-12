#include <core/GameServerProxy.hpp>

#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <core/Constants.hpp>
#include <core/Packet.hpp>
#include <util/Exception.hpp>
#include <util/LoggingUtil.hpp>

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
GameServerProxy<Game>::SharedData::SharedData(const Params& params) : params_(params) {
  util::clean_assert(params_.remote_port > 0, "Remote port must be specified");
  socket_ = io::Socket::create_client_socket(params_.remote_server, params_.remote_port);
  std::cout << "Connected to the server!" << std::endl;
}

template <concepts::Game Game>
GameServerProxy<Game>::SharedData::~SharedData() {
  for (auto gen : players_) {
    delete gen;
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::register_player(seat_index_t seat,
                                                             PlayerGenerator* gen) {
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
    seat_generator_t& seat_generator = seat_generators_[i];
    seat_index_t seat = seat_generator.seat;
    PlayerGenerator* gen = seat_generator.gen;

    std::string registered_name = gen->get_name();
    if (registered_name.empty()) {
      registered_name = gen->get_default_name();
    }

    printf("Registering player \"%s\" at seat %d\n", registered_name.c_str(), seat);
    std::cout.flush();

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
    printf("Registered player \"%s\" at seat %d (player_id:%d)\n", name.c_str(), seat,
           (int)player_id);
    std::cout.flush();
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::end_session() {
  for (auto& sg : seat_generators_) {
    sg.gen->end_session();
  }
}

template <concepts::Game Game>
GameServerProxy<Game>::PlayerThread::PlayerThread(SharedData& shared_data, Player* player,
                                                       game_thread_id_t game_thread_id,
                                                       player_id_t player_id)
    : shared_data_(shared_data),
      player_(player),
      game_thread_id_(game_thread_id),
      player_id_(player_id) {
  thread_ = new std::thread([&] { run(); });
}

template <concepts::Game Game>
GameServerProxy<Game>::PlayerThread::~PlayerThread() {
  delete thread_;
  delete player_;
}

template <concepts::Game Game>
void GameServerProxy<Game>::PlayerThread::handle_start_game(const StartGame& payload) {
  if (kEnableDebug) {
    LOG_INFO << __func__ << "() game_thread:" << game_thread_id_ << " player:" << player_id_;
  }
  game_id_t game_id = payload.game_id;
  player_name_array_t player_names;
  seat_index_t seat_assignment = payload.seat_assignment;
  payload.parse_player_names(player_names);

  new (&state_) FullState();  // placement-new
  player_->init_game(game_id, player_names, seat_assignment);
  player_->start_game();
}

template <concepts::Game Game>
void GameServerProxy<Game>::PlayerThread::handle_state_change(const StateChange& payload) {
  if (kEnableDebug) {
    LOG_INFO << __func__ << "() game_thread:" << game_thread_id_ << " player:" << player_id_;
  }
  const char* buf = payload.dynamic_size_section.buf;

  seat_index_t seat = Rules::get_current_player(state_);
  ActionResponse action_response = *reinterpret_cast<const ActionResponse*>(buf);
  action_t action = action_response.action;
  Rules::apply(state_, action);

  player_->receive_state_change(seat, state_, action);
}

template <concepts::Game Game>
void GameServerProxy<Game>::PlayerThread::handle_action_prompt(const ActionPrompt& payload) {
  if (kEnableDebug) {
    LOG_INFO << __func__ << "() game_thread:" << game_thread_id_ << " player:" << player_id_;
  }
  const char* buf = payload.dynamic_size_section.buf;

  std::unique_lock lock(mutex_);
  valid_actions_ = *reinterpret_cast<const ActionMask*>(buf);
  ready_to_get_action_ = true;
  lock.unlock();
  cv_.notify_one();
}

template <concepts::Game Game>
void GameServerProxy<Game>::PlayerThread::handle_end_game(const EndGame& payload) {
  if (kEnableDebug) {
    LOG_INFO << __func__ << "() game_thread:" << game_thread_id_ << " player:" << player_id_;
  }
  const char* buf = payload.dynamic_size_section.buf;

  ValueArray outcome = *reinterpret_cast<const ValueArray*>(buf);
  player_->end_game(state_, outcome);
}

template <concepts::Game Game>
void GameServerProxy<Game>::PlayerThread::stop() {
  std::unique_lock lock(mutex_);
  active_ = false;
  lock.unlock();
  cv_.notify_one();
}

template <concepts::Game Game>
void GameServerProxy<Game>::PlayerThread::send_action_packet(const ActionResponse& response) {
  Packet<ActionDecision> packet;
  ActionDecision& decision = packet.payload();
  auto& section = decision.dynamic_size_section;

  decision.game_thread_id = game_thread_id_;
  decision.player_id = player_id_;
  packet.set_dynamic_section_size(sizeof(response));
  memcpy(section.buf, &response, sizeof(response));
  packet.send_to(shared_data_.socket());
}

template <concepts::Game Game>
void GameServerProxy<Game>::PlayerThread::run() {
  while (true) {
    std::unique_lock lock(mutex_);
    if (kEnableDebug) {
      LOG_INFO << __func__ << "() loop game_thread:" << game_thread_id_ << " player:" << player_id_;
    }
    cv_.wait(lock, [&] { return !active_ || ready_to_get_action_; });
    if (!active_) break;

    ready_to_get_action_ = false;
    send_action_packet(player_->get_action_response(state_, valid_actions_));
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::run() {
  shared_data_.init_socket();
  init_player_threads();

  while (true) {
    GeneralPacket response_packet;
    if (!response_packet.read_from(shared_data_.socket())) {
      break;
    }

    auto type = response_packet.header().type;
    switch (type) {
      case PacketHeader::kStartGame:
        handle_start_game(response_packet);
        break;
      case PacketHeader::kStateChange:
        handle_state_change(response_packet);
        break;
      case PacketHeader::kActionPrompt:
        handle_action_prompt(response_packet);
        break;
      case PacketHeader::kEndGame:
        handle_end_game(response_packet);
        break;
      default:
        throw util::Exception("Unexpected packet type: %d", (int)type);
    }
  }

  destroy_player_threads();
  shared_data_.end_session();
  util::KeyValueDumper::flush();
}

template <concepts::Game Game>
GameServerProxy<Game>::~GameServerProxy() {
  for (auto& array : thread_vec_) {
    for (auto& thread : array) {
      delete thread;
    }
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::init_player_threads() {
  Packet<GameThreadInitialization> recv_packet;
  if (!recv_packet.read_from(shared_data_.socket())) {
    throw util::Exception("Unexpected socket close");
  }
  int num_game_threads = recv_packet.payload().num_game_threads;

  for (game_thread_id_t g = 0; g < (game_thread_id_t)num_game_threads; ++g) {
    thread_array_t& array = thread_vec_.emplace_back();
    for (player_id_t p = 0; p < (player_id_t)kNumPlayers; ++p) {
      array[p] = nullptr;
      PlayerGenerator* gen = shared_data_.get_gen(p);
      if (gen) {
        Player* player = gen->generate(g);
        array[p] = new PlayerThread(shared_data_, player, g, p);
      }
    }
  }

  Packet<GameThreadInitializationResponse> send_packet;
  send_packet.send_to(shared_data_.socket());
}

template <concepts::Game Game>
void GameServerProxy<Game>::destroy_player_threads() {
  for (auto& array : thread_vec_) {
    for (auto& thread : array) {
      if (thread) {
        thread->stop();
        thread->join();
      }
    }
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::handle_start_game(const GeneralPacket& packet) {
  const StartGame& payload = packet.payload_as<StartGame>();
  thread_vec_[payload.game_thread_id][payload.player_id]->handle_start_game(payload);
}

template <concepts::Game Game>
void GameServerProxy<Game>::handle_state_change(const GeneralPacket& packet) {
  const StateChange& payload = packet.payload_as<StateChange>();
  thread_vec_[payload.game_thread_id][payload.player_id]->handle_state_change(payload);
}

template <concepts::Game Game>
void GameServerProxy<Game>::handle_action_prompt(const GeneralPacket& packet) {
  const ActionPrompt& payload = packet.payload_as<ActionPrompt>();
  thread_vec_[payload.game_thread_id][payload.player_id]->handle_action_prompt(payload);
}

template <concepts::Game Game>
void GameServerProxy<Game>::handle_end_game(const GeneralPacket& packet) {
  const EndGame& payload = packet.payload_as<EndGame>();
  thread_vec_[payload.game_thread_id][payload.player_id]->handle_end_game(payload);
}

}  // namespace core
