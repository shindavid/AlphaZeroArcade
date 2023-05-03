#include <common/GameServer.hpp>

#include <arpa/inet.h>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>

#include <boost/program_options.hpp>

#include <common/DerivedTypes.hpp>
#include <common/Packet.hpp>
#include <common/players/RemotePlayerProxy.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/Exception.hpp>
#include <util/ParamDumper.hpp>
#include <util/Random.hpp>
#include <util/SocketUtil.hpp>
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
          "max num games to play simultaneously. Participating players can request lower values and the server will respect their requests")
      .template add_bool_switches<"display-progress-bar", "hide-progress-bar">(
          &display_progress_bar, "display progress bar (only in tty-mode without TUI player)", "hide progress bar")
      ;
}

template<GameStateConcept GameState>
GameServer<GameState>::SharedData::~SharedData() {
  if (bar_) delete bar_;

  for (auto& reg : registrations_) {
    delete reg.gen;
  }
}

template<GameStateConcept GameState>
void GameServer<GameState>::SharedData::init_progress_bar() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (bar_) return;

  if (params_.display_progress_bar && params_.num_games > 0 && util::tty_mode()) {
    bar_ = new progressbar(params_.num_games + 1);  // + 1 for first update
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
  for (seat_index_t s = 0; s < kNumPlayers; ++s) {
    results_array_[s][outcome[s]]++;
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
void GameServer<GameState>::SharedData::end_session() {
  for (auto& reg : registrations_) {
    reg.gen->end_session();
  }
}

template<GameStateConcept GameState>
bool GameServer<GameState>::SharedData::ready_to_start() const {
  for (const auto& reg : registrations_) {
    auto* remote_gen = dynamic_cast<RemotePlayerProxyGenerator*>(reg.gen);
    if (remote_gen && !remote_gen->initialized()) return false;
  }
  return true;
}

template<GameStateConcept GameState>
int GameServer<GameState>::SharedData::compute_parallelism_factor() const {
  int parallelism = params_.parallelism;
  if (params_.num_games > 0) {
    parallelism = std::min(parallelism, params_.num_games);
  }

  for (const auto& reg : registrations_) {
    int n = reg.gen->max_simultaneous_games();
    if (n > 0) parallelism = std::min(parallelism, n);
  }
  return parallelism;
}

template<GameStateConcept GameState>
void GameServer<GameState>::SharedData::register_player(
    seat_index_t seat, PlayerGenerator* gen, bool implicit_remote) {
  util::clean_assert(seat < kNumPlayers, "Invalid seat number %d", seat);
  if (dynamic_cast<RemotePlayerProxyGenerator*>(gen)) {
    if (implicit_remote) {
      util::clean_assert(params_.port > 0,
                         "If specifying fewer than %d --player's, the remaining players are assumed to be remote "
                         "players. In this case, --port must be specified, so that the remote players can connect",
                         kNumPlayers);
    } else {
      util::clean_assert(params_.port > 0, "Cannot use remote players without setting --port");
    }

    util::clean_assert(seat < 0, "Cannot specify --seat with --type=Remote");
  }
  if (seat >= 0) {
    for (const auto& reg : registrations_) {
      util::clean_assert(reg.seat != seat, "Double-seated player at seat %d", seat);
    }
  }
  player_id_t player_id = registrations_.size();
  util::clean_assert(player_id < kNumPlayers, "Too many players registered (max %d)", kNumPlayers);
  std::string name = gen->get_name();
  if (name.empty()) {
    name = util::create_string("P%d", player_id);
    gen->set_name(name);
  }
  registrations_.emplace_back(gen, seat, player_id);
}

template<GameStateConcept GameState>
void GameServer<GameState>::SharedData::init_random_seat_indices() {
  for (registration_t& reg : registrations_) {
    if (reg.seat < 0) {
      random_seat_indices_[num_random_seats_++] = reg.player_id;
    }
  }
  util::Random::shuffle(&random_seat_indices_[0], &random_seat_indices_[num_random_seats_]);
}

template<GameStateConcept GameState>
typename GameServer<GameState>::player_instantiation_array_t
GameServer<GameState>::SharedData::generate_player_order(const player_instantiation_array_t &instantiations) {
  std::unique_lock lock(mutex_);
  std::next_permutation(random_seat_indices_.begin(), random_seat_indices_.begin() + num_random_seats_);
  player_id_array_t random_seat_index_permutation = random_seat_indices_;
  lock.unlock();

  player_instantiation_array_t player_order;

  int r = 0;
  for (int p = 0; p < kNumPlayers; ++p) {
    if (registrations_[p].seat < 0) {
      player_order[p] = instantiations[random_seat_index_permutation[r++]];
      util::clean_assert(player_order[p].seat < 0, "unexpected bug (p=%d, seat=%d)", p, (int)player_order[p].seat);
    } else {
      player_order[p] = instantiations[p];
      util::clean_assert(player_order[p].seat >= 0, "unexpected bug (p=%d, seat=%d)", p, (int)player_order[p].seat);
    }
    player_order[p].seat = p;
  }

  return player_order;
}

template<GameStateConcept GameState>
GameServer<GameState>::GameThread::GameThread(SharedData& shared_data, game_thread_id_t id)
: shared_data_(shared_data)
, id_(id) {
  std::bitset<kNumPlayers> human_tui_indices;
  for (int p = 0; p < kNumPlayers; ++p) {
    instantiations_[p] = shared_data_.registration_templates()[p].instantiate(id);
    human_tui_indices[p] = instantiations_[p].player->is_human_tui_player();
  }

  for (int p = 0; p < kNumPlayers; ++p) {
    std::bitset<kNumPlayers> human_tui_indices_copy(human_tui_indices);
    human_tui_indices_copy[p] = false;
    if (human_tui_indices_copy.any()) {
      instantiations_[p].player->set_facing_human_tui_player();
    }
  }

  if (!human_tui_indices.any()) {
    shared_data_.init_progress_bar();
  }
}

template<GameStateConcept GameState>
GameServer<GameState>::GameThread::~GameThread() {
  if (thread_) delete thread_;

  for (const auto& reg : instantiations_) delete reg.player;
}

template<GameStateConcept GameState>
void GameServer<GameState>::GameThread::launch() {
  thread_ = new std::thread([&] { run(); });
}

template<GameStateConcept GameState>
void GameServer<GameState>::GameThread::run() {
  const Params& params = shared_data_.params();

  while (true) {
    if (!shared_data_.request_game(params.num_games)) return;

    player_instantiation_array_t player_order = shared_data_.generate_player_order(instantiations_);

    player_array_t players;
    for (int p = 0; p < kNumPlayers; ++p) {
      players[p] = player_order[p].player;
    }

    time_point_t t1 = std::chrono::steady_clock::now();
    GameOutcome outcome = play_game(players);
    time_point_t t2 = std::chrono::steady_clock::now();

    // reindex outcome according to player_id
    GameOutcome reindexed_outcome;
    for (int p = 0; p < kNumPlayers; ++p) {
      reindexed_outcome[player_order[p].player_id] = outcome[p];
    }
    duration_t duration = t2 - t1;
    int64_t ns = duration.count();
    shared_data_.update(reindexed_outcome, ns);
  }
}

template<GameStateConcept GameState>
typename GameServer<GameState>::GameOutcome
GameServer<GameState>::GameThread::play_game(player_array_t& players) {
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
    seat_index_t seat = state.get_current_player();
    Player* player = players[seat];
    auto valid_actions = state.get_valid_actions();
    action_index_t action = player->get_action(state, valid_actions);
    if (action < 0 || action >= (int)valid_actions.size() || !valid_actions[action]) {
      // TODO: gracefully handle and prompt for retry. Otherwise, a malicious remote process can crash the server.
      throw util::Exception("Player %d (%s) attempted an illegal action (%d)", seat, player->get_name().c_str(), action);
    }
    auto outcome = state.apply_move(action);
    for (auto player2 : players) {
      player2->receive_state_change(seat, state, action);
    }
    if (is_terminal_outcome(outcome)) {
      for (auto player2 : players) {
        player2->end_game(state, outcome);
      }
      return outcome;
    }
  }

  throw std::runtime_error("should not get here");
}

template<GameStateConcept GameState>
GameServer<GameState>::GameServer(const Params& params) : shared_data_(params) {}

template<GameStateConcept GameState>
void GameServer<GameState>::wait_for_remote_player_registrations() {
  util::clean_assert(num_registered_players() <= kNumPlayers, "Invalid number of players registered: %d",
                     num_registered_players());

  // fill in missing slots with remote players
  int shortage = kNumPlayers - num_registered_players();
  for (int i = 0; i < shortage; ++i) {
    RemotePlayerProxyGenerator* gen = new RemotePlayerProxyGenerator();
    shared_data_.register_player(-1, gen, true);
  }

  std::vector<registration_t*> remote_player_registrations;
  for (int r = 0; r < shared_data_.num_registrations(); ++r) {
    registration_t& reg = shared_data_.registration_templates()[r];
    if (dynamic_cast<RemotePlayerProxyGenerator*>(reg.gen)) {
      util::clean_assert(reg.seat < 0, "Cannot specify --seat= when using --type=Remote");
      remote_player_registrations.push_back(&reg);
    }
  }

  if (remote_player_registrations.empty()) return;

  int port = get_port();
  util::clean_assert(port > 0, "Invalid port number %d", port);

  io::Socket* server_socket = io::Socket::create_server_socket(port, kNumPlayers);

  std::cout << "Waiting for remote player registrations..." << std::endl;
  int n = remote_player_registrations.size();
  int r = 0;
  while (r < n) {
    io::Socket* socket = server_socket->accept();

    int remaining_requests = 1;
    do {
      Packet<Registration> packet;
      packet.read_from(socket);  // TODO: catch exception and engage in retry-protocol with client
      const Registration &registration = packet.payload();
      std::string registered_name = registration.dynamic_size_section.player_name;
      remaining_requests = registration.remaining_requests;
      int max_simultaneous_games = registration.max_simultaneous_games;
      seat_index_t seat = registration.requested_seat;

      auto reg = remote_player_registrations[r++];
      if (!registered_name.empty()) {
        reg->gen->set_name(registered_name);
      }
      std::string name = reg->gen->get_name();
      util::clean_assert(!name.empty(), "Unexpected empty name");

      RemotePlayerProxyGenerator *gen = dynamic_cast<RemotePlayerProxyGenerator *>(reg->gen);
      gen->initialize(socket, max_simultaneous_games, reg->player_id);
      reg->seat = seat;

      Packet<RegistrationResponse> response_packet;
      response_packet.payload().player_id = reg->player_id;
      response_packet.set_player_name(name);
      response_packet.send_to(socket);

      printf("Registered player: \"%s\" (seat: %d, remaining: %d)\n", name.c_str(), (int) seat, remaining_requests);
      std::cout.flush();
    } while (remaining_requests);
  }
}

template<GameStateConcept GameState>
void GameServer<GameState>::run() {
  wait_for_remote_player_registrations();
  shared_data_.init_random_seat_indices();
  util::clean_assert(shared_data_.ready_to_start(), "Game not ready to start");

  int parallelism = shared_data_.compute_parallelism_factor();
  std::vector<GameThread*> threads;
  for (int p = 0; p < parallelism; ++p) {
    GameThread* thread = new GameThread(shared_data_, (int)threads.size());
    threads.push_back(thread);
  }

  RemotePlayerProxy<GameState>::PacketDispatcher::start_all(parallelism);
  time_point_t t1 = std::chrono::steady_clock::now();

  for (auto thread : threads) {
    thread->launch();
  }

  for (auto thread : threads) {
    thread->join();
  }

  int num_games = shared_data_.num_games_started();
  time_point_t t2 = std::chrono::steady_clock::now();
  duration_t duration = t2 - t1;
  int64_t ns = duration.count();

  results_array_t results = shared_data_.get_results();

  printf("\nAll games complete!\n");
  for (player_id_t p = 0; p < kNumPlayers; ++p) {
    printf("pid=%d name=%s %s\n", p, shared_data_.get_player_name(p).c_str(), get_results_str(results[p]).c_str());
  }
  util::ParamDumper::add("Parallelism factor", "%d", (int)threads.size());
  util::ParamDumper::add("Num games", "%d", num_games);
  util::ParamDumper::add("Total runtime", "%.3fs", ns*1e-9);
  util::ParamDumper::add("Avg runtime", "%.3fs", ns*1e-9 / num_games);

  for (auto thread: threads) {
    delete thread;
  }

  shared_data_.end_session();
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
