#include <common/GameServer.hpp>

#include <arpa/inet.h>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>

#include <boost/program_options.hpp>

#include <common/DerivedTypes.hpp>
#include <common/Packet.hpp>
#include <common/RemotePlayerProxy.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/Exception.hpp>
#include <util/ParamDumper.hpp>
#include <util/Random.hpp>
#include <util/StringUtil.hpp>

namespace common {

template<GameStateConcept GameState>
auto GameServer<GameState>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("GameServer options");

  return desc
      .template add_option<"port">(po::value<int>(&port)->default_value(port),
          "port for external players to connect to (must be set to a nonzero value if using external players)")
      .template add_option<"num-games", 'G'>(po::value<int>(&num_games)->default_value(num_games),
                                             "num games (<=0 means run indefinitely)")
      .template add_option<"parallelism", 'p'>(po::value<int>(&parallelism)->default_value(parallelism),
                                               "num games to play simultaneously")
      .template add_bool_switches<"display-progress-bar", "hide-progress-bar">(
          &display_progress_bar, "display progress bar (only used in tty mode)", "hide progress bar")
      ;
}

template<GameStateConcept GameState>
GameServer<GameState>::SharedData::SharedData(const Params& params) {
  if (params.display_progress_bar && params.num_games > 0 && util::tty_mode()) {
    bar_ = new progressbar(params.num_games + 1);  // + 1 for first update
    bar_->update();  // so that progress-bar displays immediately
  }
}

template<GameStateConcept GameState>
bool GameServer<GameState>::SharedData::request_game(int num_games) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (num_games > 0 && num_games_started_ >= num_games) return false;
  num_games_started_++;
  return true;
}

template<GameStateConcept GameState>
void GameServer<GameState>::SharedData::update(const GameOutcome& outcome, int64_t ns) {
  std::lock_guard<std::mutex> guard(mutex_);
  for (player_index_t p = 0; p < kNumPlayers; ++p) {
    results_array_[p][outcome[p]]++;
  }

  total_ns_ += ns;
  min_ns_ = std::min(min_ns_, ns);
  max_ns_ = std::max(max_ns_, ns);
  if (bar_) bar_->update();
}

template<GameStateConcept GameState>
auto GameServer<GameState>::SharedData::get_results() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return results_array_;
}

template<GameStateConcept GameState>
typename GameServer<GameState>::player_id_t
GameServer<GameState>::SharedData::register_player(player_index_t seat, PlayerGenerator* gen) {
  if (seat >= kNumPlayers) {
    throw util::Exception("Invalid seat number %d >= %d", seat, kNumPlayers);
  }
  if (seat >= 0) {
    for (int r = 0; r < num_registrations_; ++r) {
      if (registration_templates_[r].seat == seat) {
        throw util::Exception("Double-seated player at seat %d", seat);
      }
    }
  }
  player_id_t player_id = num_registrations_;
  num_registrations_++;
  registration_templates_[player_id] = registration_template_t{gen, seat, player_id};
  return player_id;
}

template<GameStateConcept GameState>
typename GameServer<GameState>::registration_array_t
GameServer<GameState>::SharedData::generate_player_order(const registration_array_t &registrations) const {
  registration_array_t player_order;

  registration_array_t random_seat_assignments;
  int num_random_assignments = 0;

  // first seat players that have dedicated seats
  for (const auto& reg : registrations) {
    if (reg.seat < 0) {
      random_seat_assignments[num_random_assignments++] = reg;
      continue;
    }
    player_order[reg.seat] = reg;
  }

  // now randomly seat players at the remaining seats
  if (num_random_assignments) {
    util::Random::shuffle(&random_seat_assignments[0], &random_seat_assignments[num_random_assignments]);
    int r = 0;
    for (int p = 0; p < kNumPlayers; ++p) {
      if (player_order[p].player) continue;
      assert(r < num_random_assignments);
      player_order[p] = random_seat_assignments[r++];
    }
  }

  for (int p = 0; p < kNumPlayers; ++p) {
    player_order[p].seat = p;
  }

  return player_order;
}

template<GameStateConcept GameState>
GameServer<GameState>::GameThread::GameThread(SharedData& shared_data)
: shared_data_(shared_data) {
  void* play_address = this;
  for (int p = 0; p < kNumPlayers; ++p) {
    registrations_[p] = shared_data_.registration_templates()[p].instantiate(play_address);
  }
}

template<GameStateConcept GameState>
GameServer<GameState>::GameThread::~GameThread() {
  if (thread_) delete thread_;

  for (const auto& reg : registrations_) delete reg.player;
}

template<GameStateConcept GameState>
void GameServer<GameState>::GameThread::launch(const Params& params) {
  thread_ = new std::thread([&] { run(params); });
}

template<GameStateConcept GameState>
void GameServer<GameState>::GameThread::run(const Params& params) {
  while (true) {
    if (!shared_data_.request_game(params.num_games)) return;
    registration_array_t player_order = shared_data_.generate_player_order(registrations_);

    player_array_t players;
    for (int p = 0; p < kNumPlayers; ++p) {
      players[p] = player_order[p].player;
    }

    time_point_t t1 = std::chrono::steady_clock::now();
    GameOutcome outcome = play_game(players);
    // reindex outcome according to player_id
    GameOutcome reindexed_outcome;
    for (int p = 0; p < kNumPlayers; ++p) {
      reindexed_outcome[player_order[p].player_id] = outcome[p];
    }
    time_point_t t2 = std::chrono::steady_clock::now();
    duration_t duration = t2 - t1;
    int64_t ns = duration.count();
    shared_data_.update(reindexed_outcome, ns);
  }
}

template<GameStateConcept GameState>
typename GameServer<GameState>::GameOutcome
GameServer<GameState>::GameThread::play_game(const player_array_t& players) {
  game_id_t game_id = util::get_unique_id();

  player_name_array_t player_names;
  for (size_t p = 0; p < players.size(); ++p) {
    player_names[p] = players[p]->get_name();
  }
  for (size_t p = 0; p < players.size(); ++p) {
    players[p]->init_game(game_id, player_names, p);
    players[p]->start_game();
  }

  GameState state;
  while (true) {
    player_index_t p = state.get_current_player();
    Player* player = players[p];
    auto valid_actions = state.get_valid_actions();
    action_index_t action = player->get_action(state, valid_actions);
    if (!valid_actions[action]) {
      throw util::Exception("Player %d (%s) attempted an illegal action (%d)", p, player->get_name().c_str(), action);
    }
    auto outcome = state.apply_move(action);
    for (auto player2 : players) {
      player2->receive_state_change(p, state, action, outcome);
    }
    if (is_terminal_outcome(outcome)) {
      return outcome;
    }
  }

  throw std::runtime_error("should not get here");
}

template<GameStateConcept GameState>
GameServer<GameState>::GameServer(const Params& params) : params_(params), shared_data_(params) {}

template<GameStateConcept GameState>
void GameServer<GameState>::wait_for_remote_player_registrations() {
  if (port() <= 0) {
    throw util::Exception("Invalid port number %d", port());
  }

  // setup socket
  sockaddr_in socket_address_info;
  bzero((char *) &socket_address_info, sizeof(socket_address_info));
  socket_address_info.sin_family = AF_INET;
  socket_address_info.sin_addr.s_addr = htonl(INADDR_ANY);
  socket_address_info.sin_port = htons(port());

  // open socket
  int server_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (server_socket < 0) {
    throw util::Exception("Error establishing server socket");
  }

  // bind socket
  int bind_status = bind(server_socket, (sockaddr*) &socket_address_info, sizeof(socket_address_info));
  if (bind_status < 0) {
    throw util::Exception("Error binding socket to local address");
  }

  if (listen(server_socket, kNumPlayers)) {
    throw util::Exception("listen() failed");
  }

  std::cout << "Waiting for remote player registrations..." << std::endl;
  while (!ready_to_start()) {
    // accept connection
    sockaddr_in new_socket_address_info;
    socklen_t new_socket_address_size = sizeof(new_socket_address_info);
    int new_socket_descr = accept(server_socket, (sockaddr *) &new_socket_address_info, &new_socket_address_size);
    if (new_socket_descr < 0) {
      throw util::Exception("Error accepting request from client!");
    }

    std::cout << "Accepted connection from " << inet_ntoa(new_socket_address_info.sin_addr) << std::endl;

    char buf[1024];
    Packet packet = Packet::from_socket(new_socket_descr, buf, sizeof(buf));
    Registration registration = packet.to_registration();
    const std::string& name = registration.player_name;
    printf("Registered player: \"%s\" (seat: %d)", name.c_str(), registration.requested_seat);
//    player_generator_t gen = [&]() { return new RemotePlayerProxy<GameState>(name, new_socket_descr); };
//    register_player(registration.requested_seat, gen);
    throw util::Exception("Not implemented");
  }
}

template<GameStateConcept GameState>
void GameServer<GameState>::run() {
  if (!ready_to_start()) {
    throw util::Exception("Cannot start game with %d players (need %d)", num_registered_players(), kNumPlayers);
  }

  int parallelism = params_.parallelism;
  for (int p = 0; p < parallelism; ++p) {
    threads_.push_back(new GameThread(shared_data_));
  }

  time_point_t t1 = std::chrono::steady_clock::now();

  for (auto thread : threads_) {
    thread->launch(params_);
  }

  for (auto thread : threads_) {
    thread->join();
  }

  int num_games = shared_data_.num_games_started();
  time_point_t t2 = std::chrono::steady_clock::now();
  duration_t duration = t2 - t1;
  int64_t ns = duration.count();

  results_array_t results = shared_data_.get_results();

  printf("\nAll games complete!\n");
  for (player_index_t p = 0; p < kNumPlayers; ++p) {
    printf("P%d %s\n", p, get_results_str(results[p]).c_str());
  }
  util::ParamDumper::add("Parallelism factor", "%d", parallelism);
  util::ParamDumper::add("Num games", "%d", num_games);
  util::ParamDumper::add("Total runtime", "%.3fs", ns*1e-9);
  util::ParamDumper::add("Avg runtime", "%.3fs", ns*1e-9 / num_games);

  for (auto thread: threads_) {
    delete thread;
  }
}

template<GameStateConcept GameState>
std::string GameServer<GameState>::get_results_str(const results_map_t& map) {
  int win = 0;
  int loss = 0;
  int draw = 0;
  float score = 0;

  for (auto it : map) {
    float f = it.first;
    int count = it.second;
    score += f * count;
    if (f == 1) win += count;
    else if (f == 0) loss += count;
    else draw += count;
  }
  return util::create_string("W%d L%d D%d [%.16g]", win, loss, draw, score);
}

}  // namespace common
