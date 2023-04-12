#include <common/GameServerProxy.hpp>

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

#include <common/Constants.hpp>
#include <common/Packet.hpp>
#include <util/Exception.hpp>

namespace common {

template <GameStateConcept GameState>
auto GameServerProxy<GameState>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Remote GameServer options");

  return desc
      .template add_option<"remote-server">(po::value<std::string>(&remote_server)->default_value(remote_server),
          "Remote server to connect players to")
      .template add_option<"remote-port">(po::value<int>(&remote_port),
          "Remote port to connect players to. If not specified, run server in-process")
      ;
}

template <GameStateConcept GameState>
GameServerProxy<GameState>::SharedData::SharedData(const Params& params)
: params_(params)
{
  util::clean_assert(params_.remote_port > 0, "Remote port must be specified");
  // setup a socket and connection tools
  struct hostent *host = gethostbyname(params_.remote_server.c_str());
  int port = params_.remote_port;

  sockaddr_in socket_address_info;
  bzero((char *) &socket_address_info, sizeof(socket_address_info));
  socket_address_info.sin_family = AF_INET;
  socket_address_info.sin_addr.s_addr =
      inet_addr(inet_ntoa(*(struct in_addr *) *host->h_addr_list));
  socket_address_info.sin_port = htons(port);

  // open socket
  socket_desc_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_desc_ < 0) {
    throw util::Exception("Error establishing client socket");
  }

  int status = connect(socket_desc_, (sockaddr *) &socket_address_info, sizeof(socket_address_info));
  if (status < 0) {
    throw util::Exception("Error connecting to socket");
  }
  std::cout << "Connected to the server!" << std::endl;
}

template <GameStateConcept GameState>
void GameServerProxy<GameState>::SharedData::register_player(seat_index_t seat, PlayerGenerator* gen) {
  std::string name = gen->get_name();
  util::clean_assert(name.size() + 1 < kMaxNameLength, "Player name too long (\"%s\" size=%d)",
                     name.c_str(), (int)name.size());

  printf("Registering player \"%s\" at seat %d\n", name.c_str(), seat);
  std::cout.flush();

  Packet<Registration> send_packet;
  Registration& registration = send_packet.payload();
  registration.requested_seat = seat;

  char* name_buf = registration.dynamic_size_section.player_name;
  size_t name_buf_size = sizeof(registration.dynamic_size_section.player_name);
  strncpy(name_buf, name.c_str(), name_buf_size);
  name_buf[name_buf_size - 1] = '\0';  // not needed because of clean_assert() above, but makes compiler happy
  send_packet.set_dynamic_section_size(name.size() + 1);  // + 1 for null-delimiter
  send_packet.send_to(socket_desc_);

  Packet<RegistrationResponse> recv_packet;
  recv_packet.read_from(socket_desc_);
  const RegistrationResponse& response = recv_packet.payload();
  player_id_t player_id = response.player_id;
  util::clean_assert(player_id >= 0 && player_id < kNumPlayers, "Invalid player_id: %d", (int)player_id);

  player_generators_[player_id] = gen;
  printf("Registered player \"%s\" at seat %d (player_id:%d)\n", name.c_str(), seat, (int)player_id);
  std::cout.flush();
}

template<GameStateConcept GameState>
GameServerProxy<GameState>::GameThread::GameThread(SharedData& shared_data, game_thread_id_t id)
: shared_data_(shared_data), id_(id)
{
  for (int p = 0; p < kNumPlayers; ++p) {
    PlayerGenerator* gen = shared_data_.get_gen(p);
    if (gen) {
      Player* player = gen->generate(id);
      players_[p] = player;
      int m = player->max_simultaneous_games();
      if (m > 0) {
        if (max_simultaneous_games_ == 0) {
          max_simultaneous_games_ = m;
        } else {
          max_simultaneous_games_ = std::min(max_simultaneous_games_, m);
        }
      }
    }
  }
}

template<GameStateConcept GameState>
GameServerProxy<GameState>::GameThread::~GameThread() {
  if (thread_) delete thread_;
}

template<GameStateConcept GameState>
void GameServerProxy<GameState>::GameThread::handle_start_game(const StartGame& payload) {
  player_id_t player_id = payload.player_id;
  game_id_t game_id = payload.game_id;
  player_name_array_t player_names;
  seat_index_t seat_assignment = payload.seat_assignment;
  payload.parse_player_names(player_names);

  Player* player = players_[player_id];
  util::clean_assert(player, "Invalid player_id: %d", (int)payload.player_id);

  std::unique_lock lock(mutex_);
  player->init_game(game_id, player_names, seat_assignment);
}

template<GameStateConcept GameState>
void GameServerProxy<GameState>::GameThread::launch() {
  thread_ = new std::thread([&] { run(); });
}

template<GameStateConcept GameState>
void GameServerProxy<GameState>::GameThread::run()
{
  std::unique_lock lock(mutex_);
  cv_.wait(lock);

  while (true) {
    // TODO: synchronization magic with mutex/cv
  }
}

template <GameStateConcept GameState>
void GameServerProxy<GameState>::run()
{
  while (true) {
    GeneralPacket response_packet;
    response_packet.read_from(shared_data_.socket_desc());

    auto type = response_packet.header().type;
    switch (type) {
      case PacketHeader::kGameThreadInitialization:
        handle_game_thread_initialization(response_packet);
        break;
      case PacketHeader::kStartGame:
        handle_start_game(response_packet);
        break;
      default:
        throw util::Exception("Unexpected packet type: %d", (int) type);
    }
  }
}

template <GameStateConcept GameState>
void GameServerProxy<GameState>::handle_game_thread_initialization(const GeneralPacket& packet) {
  const GameThreadInitialization& payload = packet.to<GameThreadInitialization>();
  GameThread* thread = new GameThread(shared_data_, payload.game_thread_id);
  util::clean_assert(payload.game_thread_id == (int)thread_vec_.size(),
                     "GameThreadInitialization packet has unexpected game_thread_id: %d (expected %d)",
                      payload.game_thread_id, (int)thread_vec_.size());
  thread_vec_.push_back(thread);
  printf("Created new GameThread (%d)\n", payload.game_thread_id);
  std::cout.flush();

  Packet<GameThreadInitializationResponse> send_packet;
  send_packet.payload().max_simultaneous_games = thread->max_simultaneous_games();
  send_packet.send_to(shared_data_.socket_desc());

  thread->launch();
}

template <GameStateConcept GameState>
void GameServerProxy<GameState>::handle_start_game(const GeneralPacket& packet) {
  const StartGame& payload = packet.to<StartGame>();

  GameThread* thread = thread_vec_[payload.game_thread_id];
  thread->handle_start_game(payload);
}

}  // namespace common
