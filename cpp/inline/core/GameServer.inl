#include <core/GameServer.hpp>

#include <arpa/inet.h>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>

#include <boost/program_options.hpp>

#include <core/Packet.hpp>
#include <core/players/RemotePlayerProxy.hpp>
#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/Exception.hpp>
#include <util/KeyValueDumper.hpp>
#include <util/Random.hpp>
#include <util/SocketUtil.hpp>
#include <util/StringUtil.hpp>

namespace core {

template <concepts::Game Game>
auto GameServer<Game>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("GameServer options");

  return desc
      .template add_option<"port">(po::value<int>(&port)->default_value(port),
                                   "port for external players to connect to (must be set to a "
                                   "nonzero value if using external players)")
      .template add_option<"num-games", 'G'>(po::value<int>(&num_games)->default_value(num_games),
                                             "num games (<=0 means run indefinitely)")
      .template add_option<"parallelism", 'p'>(
          po::value<int>(&parallelism)->default_value(parallelism),
          "max num games to play simultaneously. Participating players can request lower values "
          "and the server will respect their requests")
      .template add_flag<"display-progress-bar", "hide-progress-bar">(
          &display_progress_bar, "display progress bar (only in tty-mode without TUI player)",
          "hide progress bar")
      .template add_flag<"print-game-states", "do-not-print-game-states">(
          &print_game_states, "print game state between moves",
          "do not print game state between moves")
      .template add_flag<"announce-game-results", "do-not-announce-game-results">(
          &announce_game_results, "announce result after each individual game",
          "do not announce result after each individual game")
      .template add_hidden_flag<"respect-victory-hints", "do-not-respect-victory-hints">(
          &respect_victory_hints, "immediately exit game if a player claims imminent victory",
          "ignore imminent victory claims from players");
}

template <concepts::Game Game>
GameServer<Game>::SharedData::~SharedData() {
  if (bar_) delete bar_;

  for (auto& reg : registrations_) {
    delete reg.gen;
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::init_progress_bar() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (bar_) return;

  if (params_.display_progress_bar && params_.num_games > 0 && util::tty_mode()) {
    bar_ = new progressbar(params_.num_games + 1);  // + 1 for first update
    bar_->update();                                 // so that progress-bar displays immediately
  }
}

template <concepts::Game Game>
bool GameServer<Game>::SharedData::request_game(int num_games) {
  if (LoopControllerClient::deactivated()) return false;
  std::lock_guard<std::mutex> guard(mutex_);
  if (num_games > 0 && num_games_started_ >= num_games) return false;
  num_games_started_++;
  return true;
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::update(const ValueArray& outcome, int64_t ns) {
  std::lock_guard<std::mutex> guard(mutex_);
  for (seat_index_t s = 0; s < kNumPlayers; ++s) {
    results_array_[s][outcome[s]]++;
  }

  total_ns_ += ns;
  min_ns_ = std::min(min_ns_, ns);
  max_ns_ = std::max(max_ns_, ns);
  if (bar_) bar_->update();
}

template <concepts::Game Game>
auto GameServer<Game>::SharedData::get_results() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return results_array_;
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::end_session() {
  for (auto& reg : registrations_) {
    reg.gen->end_session();
  }
}

template <concepts::Game Game>
bool GameServer<Game>::SharedData::ready_to_start() const {
  for (const auto& reg : registrations_) {
    auto* remote_gen = dynamic_cast<RemotePlayerProxyGenerator*>(reg.gen);
    if (remote_gen && !remote_gen->initialized()) return false;
  }
  return true;
}

template <concepts::Game Game>
int GameServer<Game>::SharedData::compute_parallelism_factor() const {
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

template <concepts::Game Game>
void GameServer<Game>::SharedData::register_player(seat_index_t seat, PlayerGenerator* gen,
                                                        bool implicit_remote) {
  util::clean_assert(seat < kNumPlayers, "Invalid seat number %d", seat);
  if (dynamic_cast<RemotePlayerProxyGenerator*>(gen)) {
    if (implicit_remote) {
      util::clean_assert(
          params_.port > 0,
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
    gen->set_name(gen->get_default_name());
  }
  registrations_.emplace_back(gen, seat, player_id);
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::init_random_seat_indices() {
  std::bitset<kNumPlayers> fixed_seat_indices;
  for (registration_t& reg : registrations_) {
    if (reg.seat >= 0) {
      fixed_seat_indices.set(reg.seat);
    }
  }

  for (seat_index_t random_seat : bitset_util::off_indices(fixed_seat_indices)) {
    random_seat_indices_[num_random_seats_++] = random_seat;
  }
  util::Random::shuffle(&random_seat_indices_[0], &random_seat_indices_[num_random_seats_]);
}

template <concepts::Game Game>
typename GameServer<Game>::player_instantiation_array_t
GameServer<Game>::SharedData::generate_player_order(
    const player_instantiation_array_t& instantiations) {
  std::unique_lock lock(mutex_);
  std::next_permutation(random_seat_indices_.begin(),
                        random_seat_indices_.begin() + num_random_seats_);
  player_id_array_t random_seat_index_permutation = random_seat_indices_;
  lock.unlock();

  player_instantiation_array_t player_order;

  for (int p = 0; p < kNumPlayers; ++p) {
    int s = registrations_[p].seat;
    if (s < 0) continue;
    player_order[s] = instantiations[p];
  }

  int r = 0;
  for (int p = 0; p < kNumPlayers; ++p) {
    int s = registrations_[p].seat;
    if (s >= 0) continue;
    s = random_seat_index_permutation[r++];
    player_order[s] = instantiations[p];
  }

  for (int p = 0; p < kNumPlayers; ++p) {
    player_order[p].seat = p;
  }

  return player_order;
}

template <concepts::Game Game>
GameServer<Game>::GameThread::GameThread(SharedData& shared_data, game_thread_id_t id)
    : shared_data_(shared_data), id_(id) {
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

template <concepts::Game Game>
GameServer<Game>::GameThread::~GameThread() {
  if (thread_) delete thread_;

  for (const auto& reg : instantiations_) delete reg.player;
}

template <concepts::Game Game>
void GameServer<Game>::GameThread::run() {
  const Params& params = shared_data_.params();

  while (!decommissioned_) {
    if (!shared_data_.request_game(params.num_games)) return;

    player_instantiation_array_t player_order = shared_data_.generate_player_order(instantiations_);

    player_array_t players;
    for (int p = 0; p < kNumPlayers; ++p) {
      players[p] = player_order[p].player;
    }

    time_point_t t1 = std::chrono::steady_clock::now();
    ValueArray outcome = play_game(players);
    time_point_t t2 = std::chrono::steady_clock::now();

    // reindex outcome according to player_id
    ValueArray reindexed_outcome;
    for (int p = 0; p < kNumPlayers; ++p) {
      reindexed_outcome[player_order[p].player_id] = outcome[p];
    }
    duration_t duration = t2 - t1;
    int64_t ns = duration.count();
    shared_data_.update(reindexed_outcome, ns);
  }
}

template <concepts::Game Game>
typename GameServer<Game>::ValueArray GameServer<Game>::GameThread::play_game(
    player_array_t& players) {
  game_id_t game_id = util::get_unique_id();

  player_name_array_t player_names;
  for (size_t p = 0; p < players.size(); ++p) {
    player_names[p] = players[p]->get_name();
  }
  for (size_t p = 0; p < players.size(); ++p) {
    players[p]->init_game(game_id, player_names, p);
    players[p]->start_game();
  }

  StateHistory state_history;
  state_history.initialize(Rules{});

  if (shared_data_.params().print_game_states) {
    Game::IO::print_state(std::cout, state_history.current(), -1, &player_names);
  }
  while (true) {
    seat_index_t seat = Rules::get_current_player(state_history.current());
    Player* player = players[seat];
    auto valid_actions = Rules::get_legal_moves(state_history);
    ActionResponse response = player->get_action_response(state_history.current(), valid_actions);
    action_t action = response.action;

    // TODO: gracefully handle and prompt for retry. Otherwise, a malicious remote process can crash
    // the server.
    util::release_assert(valid_actions[action], "Invalid action: %d", action);

    ActionOutcome outcome;
    if (response.victory_guarantee && shared_data_.params().respect_victory_hints) {
      outcome.terminal_value.setZero();
      outcome.terminal_value[seat] = 1;
      if (shared_data_.params().announce_game_results) {
        printf("Short-circuiting game %ld because player %s (seat=%d) claims victory\n", game_id,
               player->get_name().c_str(), int(seat));
        std::cout << std::endl;
      }
    } else {
      outcome = Rules::apply(state_history, action);
      if (shared_data_.params().print_game_states) {
        Game::IO::print_state(std::cout, state_history.current(), action, &player_names);
      }
      for (auto player2 : players) {
        player2->receive_state_change(seat, state_history.current(), action);
      }
    }
    if (outcome.terminal) {
      for (auto player2 : players) {
        player2->end_game(state_history.current(), outcome.terminal_value);
      }
      if (shared_data_.params().announce_game_results) {
        printf("Game %ld complete.\n", game_id);
        for (player_id_t p = 0; p < kNumPlayers; ++p) {
          printf("  pid=%d name=%s %g\n", p, players[p]->get_name().c_str(),
                 outcome.terminal_value[p]);
        }
        std::cout << std::endl;
      }
      return outcome.terminal_value;
    }
  }

  throw std::runtime_error("should not get here");
}

template <concepts::Game Game>
GameServer<Game>::GameServer(const Params& params) : shared_data_(params) {}

template <concepts::Game Game>
void GameServer<Game>::wait_for_remote_player_registrations() {
  util::clean_assert(num_registered_players() <= kNumPlayers,
                     "Invalid number of players registered: %d", num_registered_players());

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
      if (!packet.read_from(socket)) {
        throw util::Exception("Unexpected socket close");
      }
      const Registration& registration = packet.payload();
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

      RemotePlayerProxyGenerator* gen = dynamic_cast<RemotePlayerProxyGenerator*>(reg->gen);
      gen->initialize(socket, max_simultaneous_games, reg->player_id);
      reg->seat = seat;

      Packet<RegistrationResponse> response_packet;
      response_packet.payload().player_id = reg->player_id;
      response_packet.set_player_name(name);
      response_packet.send_to(socket);

      printf("Registered player: \"%s\" (seat: %d, remaining: %d)\n", name.c_str(), (int)seat,
             remaining_requests);
      std::cout.flush();
    } while (remaining_requests);
  }
}

template <concepts::Game Game>
void GameServer<Game>::run() {
  wait_for_remote_player_registrations();
  shared_data_.init_random_seat_indices();
  util::clean_assert(shared_data_.ready_to_start(), "Game not ready to start");

  int parallelism = shared_data_.compute_parallelism_factor();
  for (int p = 0; p < parallelism; ++p) {
    GameThread* thread = new GameThread(shared_data_, (int)threads_.size());
    threads_.push_back(thread);
  }

  RemotePlayerProxy<Game>::PacketDispatcher::start_all(parallelism);

  time_point_t start_time = std::chrono::steady_clock::now();
  for (auto thread : threads_) {
    thread->launch();
  }
  for (auto thread : threads_) {
    thread->join();
  }
  time_point_t end_time = std::chrono::steady_clock::now();

  int num_games = shared_data_.num_games_started();
  duration_t duration = end_time - start_time;
  int64_t ns = duration.count();

  results_array_t results = shared_data_.get_results();

  fprintf(stderr, "\n");  // flush progress-bar
  LOG_INFO << "All games complete!";
  for (player_id_t p = 0; p < kNumPlayers; ++p) {
    LOG_INFO << util::create_string("pid=%d name=%s %s", p, shared_data_.get_player_name(p).c_str(),
                                    get_results_str(results[p]).c_str());
  }

  util::KeyValueDumper::add("Parallelism factor", "%d", (int)threads_.size());
  util::KeyValueDumper::add("Num games", "%d", num_games);
  util::KeyValueDumper::add("Total runtime", "%.3fs", ns * 1e-9);
  util::KeyValueDumper::add("Avg runtime", "%.3fs", ns * 1e-9 / num_games);

  for (auto thread : threads_) {
    delete thread;
  }

  shared_data_.end_session();
  util::KeyValueDumper::flush();
}

template <concepts::Game Game>
std::string GameServer<Game>::get_results_str(const results_map_t& map) {
  int win = 0;
  int loss = 0;
  int draw = 0;
  float score = 0;

  for (auto it : map) {
    float f = it.first;
    int count = it.second;
    score += f * count;
    if (f == 1)
      win += count;
    else if (f == 0)
      loss += count;
    else
      draw += count;
  }
  return util::create_string("W%d L%d D%d [%.16g]", win, loss, draw, score);
}

}  // namespace core
