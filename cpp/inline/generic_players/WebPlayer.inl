#include "generic_players/WebPlayer.hpp"

#include "core/BasicTypes.hpp"
#include "util/Asserts.hpp"
#include "util/BitSet.hpp"
#include "util/Exceptions.hpp"
#include "util/LoggingUtil.hpp"
#include "util/OsUtil.hpp"
#include "util/StringUtil.hpp"

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <string>

namespace generic {

template <core::concepts::Game Game>
WebPlayer<Game>::WebPlayer() : acceptor_(create_acceptor()), socket_(io_context_) {
  thread_ = mit::thread([this]() { response_loop(); });
}

template <core::concepts::Game Game>
WebPlayer<Game>::~WebPlayer() {
  if (bridge_process_) {
    bridge_process_->terminate();
    delete bridge_process_;
  }
  if (frontend_process_) {
    frontend_process_->terminate();
    delete frontend_process_;
  }
}

template <core::concepts::Game Game>
void WebPlayer<Game>::start_game() {
  last_action_ = -1;
  last_mode_ = 0;
  action_ = -1;
  ready_for_response_ = false;
  resign_ = false;

  if (first_game_) {
    // sleep for one second to ensure response_loop() is ready
    std::this_thread::sleep_for(std::chrono::seconds(1));
    launch_bridge();
    launch_frontend();

    std::cout << "Please open the frontend in your browser at:\n\n"
              << "    http://localhost:5173\n"
              << std::endl;

    {
      mit::unique_lock lock(mutex_);
      cv_.wait(lock, [this]() { return connected_; });
    }

    first_game_ = false;
    return;
  }

  // Wait for player to click "Start Game" in the frontend
  while (true) {
    RELEASE_ASSERT(connected_);
    boost::asio::streambuf buf;
    boost::asio::read_until(socket_, buf, '\n');
    std::istream is(&buf);
    std::string line;
    std::getline(is, line);
    auto parsed = boost::json::parse(line);
    const auto& obj = parsed.as_object();
    std::string type = obj.at("type").as_string().c_str();
    if (type == "new_game") {
      break;
    } else {
      throw util::Exception("Expected 'new_game' type, got: {}", type);
    }
  }
}

template <core::concepts::Game Game>
void WebPlayer<Game>::receive_state_change(core::seat_index_t seat, const State& state,
                                           core::action_t action) {
  last_action_ = action;
  last_mode_ = Game::Rules::get_action_mode(state);
}

template <core::concepts::Game Game>
typename WebPlayer<Game>::ActionResponse WebPlayer<Game>::get_action_response(
  const ActionRequest& request) {
  if (resign_) {
    return ActionResponse::resign();
  }
  if (action_ != -1) {
    core::action_t action = action_;
    action_ = -1;
    return action;
  }
  send_state(request.state, last_action_, last_mode_);
  notification_unit_ = request.notification_unit;

  // NOTE: see long comment in GameServer.inl, in the next() method, about a race condition
  // consideration about this.
  mit::unique_lock lock(mutex_);
  ready_for_response_ = true;
  lock.unlock();
  cv_.notify_all();

  return ActionResponse::yield();
}

template <core::concepts::Game Game>
void WebPlayer<Game>::end_game(const State& state, const ValueTensor& outcome) {
  boost::json::object msg;
  msg["type"] = "game_end";
  boost::json::object payload = make_state_msg(state, last_action_, last_mode_);
  payload["msg"] = make_result_msg(state, outcome);
  msg["payload"] = payload;
  std::string out = boost::json::serialize(msg) + "\n";
  RELEASE_ASSERT(connected_);
  boost::asio::write(socket_, boost::asio::buffer(out));
}

template <core::concepts::Game Game>
std::string WebPlayer<Game>::make_result_msg(const State& state, const ValueTensor& outcome) {
  constexpr int P = Game::Constants::kNumPlayers;

  auto array = Game::GameResults::to_value_array(outcome);

  std::bitset<P> winners;
  for (int p = 0; p < P; ++p) {
    winners[p] = array[p] > 0;
  }

  if (winners.count() > 1) {
    if (P == 2) {
      // In a 2-player game, no need to say the players' names, just say "Draw!"
      return "Draw!";
    }

    std::vector<std::string> winner_names;
    for (int p = 0; p < P; ++p) {
      if (winners[p]) {
        winner_names.push_back(IO::player_to_str(p));
      }
    }

    return std::format("Draw between {}", util::grammatically_join(winner_names, "and"));
  } else if (winners.count() == 1) {
    core::seat_index_t w = bitset_util::choose_random_on_index(winners);
    return IO::player_to_str(w) + " wins!";
  } else {
    throw util::Exception("Game ended with no winners!");
  }
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_state_msg(const State& state, core::action_t last_action,
                                                    core::action_mode_t last_mode) {
  boost::json::object payload;
  payload["board"] = IO::compact_state_repr(state);
  payload["last_action"] = IO::action_to_str(last_action, last_mode);
  payload["turn"] = IO::player_to_str(this->get_my_seat());
  return payload;
}

template <core::concepts::Game Game>
void WebPlayer<Game>::launch_bridge() {
  namespace bp = boost::process;
  namespace bf = boost::filesystem;

  os_util::free_port(bridge_port_);

  bp::environment env = boost::this_process::environment();
  env["BRIDGE_PORT"] = std::to_string(bridge_port_);
  env["ENGINE_PORT"] = std::to_string(engine_port_);
  env["SPAWN_ENGINE"] = "false";

  bf::path start_dir = "/workspace/repo/web";
  bf::path log_dir = std::format("/home/devuser/scratch/logs/{}", Game::Constants::kGameName);
  bf::create_directories(log_dir);
  bf::path log_file = log_dir / "bridge.log";

  std::string cmd = "npm run bridge";

  bridge_process_ = new bp::child(cmd, bp::start_dir = start_dir, bp::std_out > log_file,
                                  bp::std_err > log_file, env);

  LOG_INFO("Web player launched bridge process on port {}", bridge_port_);
}

template <core::concepts::Game Game>
void WebPlayer<Game>::launch_frontend() {
  namespace bp = boost::process;
  namespace bf = boost::filesystem;

  os_util::free_port(vite_port_);

  bp::environment env = boost::this_process::environment();
  env["VITE_BRIDGE_PORT"] = std::to_string(bridge_port_);

  bf::path start_dir = "/workspace/repo/web";
  bf::path log_dir = std::format("/home/devuser/scratch/logs/{}", Game::Constants::kGameName);
  bf::create_directories(log_dir);
  bf::path log_file = log_dir / "frontend.log";

  std::string cmd = std::format("npm --workspace=games/{} run dev", Game::Constants::kGameName);

  frontend_process_ = new bp::child(cmd, bp::start_dir = start_dir, bp::std_out > log_file,
                                    bp::std_err > log_file, env);

  LOG_INFO("Web player launched frontend process");
}

template <core::concepts::Game Game>
void WebPlayer<Game>::response_loop() {
  LOG_INFO("Web player is waiting for a connection on port {}", engine_port_);
  acceptor_.accept(socket_);
  LOG_INFO("Web player connected to client.");

  {
    mit::unique_lock lock(mutex_);
    connected_ = true;
  }
  cv_.notify_all();

  // TODO: wire end_session() to WebPlayer, and use that to break this loop
  // And/or: rely on socket_.is_open()
  while (true) {
    {
      mit::unique_lock lock(mutex_);
      cv_.wait(lock, [this]() { return ready_for_response_; });
      ready_for_response_ = false;
    }

    boost::asio::streambuf buf;
    boost::asio::read_until(socket_, buf, '\n');
    std::istream is(&buf);
    std::string line;
    std::getline(is, line);
    auto parsed = boost::json::parse(line);
    const auto& obj = parsed.as_object();
    std::string type = obj.at("type").as_string().c_str();
    if (type == "make_move") {
      const auto& payload = obj.at("payload").as_object();
      int idx = payload.at("index").as_int64();

      // TODO: extend Game::IO interface to have a str_to_action() method - that gives us the
      // flexibility to use "index" or "action" in the JSON.
      action_ = idx;
    } else if (type == "resign") {
      resign_ = true;
    } else {
      throw util::Exception("Unknown message type: {}", type);
    }
    notification_unit_.yield_manager->notify(notification_unit_);
  }
}

// NOTE: we cannot send anything to socket_ until after acceptor_.accept() is called in
// response_loop(). If we hit this method before that point, we need to store the state, and send it
// later.
template <core::concepts::Game Game>
void WebPlayer<Game>::send_state(const State& state, core::action_t last_action,
                                 core::action_mode_t last_mode) {
  mit::unique_lock lock(mutex_);
  RELEASE_ASSERT(connected_);
  write_to_socket(state, last_action, last_mode);
}

template <core::concepts::Game Game>
void WebPlayer<Game>::write_to_socket(const State& state, core::action_t last_action,
                                      core::action_mode_t last_mode) {
  boost::json::object msg;
  msg["type"] = "state_update";
  msg["payload"] = make_state_msg(state, last_action, last_mode);
  std::string out = boost::json::serialize(msg) + "\n";
  boost::asio::write(socket_, boost::asio::buffer(out));
}

template <core::concepts::Game Game>
boost::asio::ip::tcp::acceptor WebPlayer<Game>::create_acceptor() {
  os_util::free_port(engine_port_);

  boost::asio::ip::tcp::acceptor acceptor(io_context_);
  acceptor.open(boost::asio::ip::tcp::v4());
  acceptor.set_option(boost::asio::socket_base::reuse_address(true));
  acceptor.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), engine_port_));
  acceptor.listen();
  return acceptor;
}

}  // namespace generic
