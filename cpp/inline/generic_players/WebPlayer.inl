#include "generic_players/WebPlayer.hpp"

#include "core/BasicTypes.hpp"
#include "util/BitSet.hpp"
#include "util/Exceptions.hpp"
#include "util/LoggingUtil.hpp"
#include "util/OsUtil.hpp"
#include "util/Rendering.hpp"

#include <boost/filesystem.hpp>
#include <boost/json/array.hpp>
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

  {
    mit::unique_lock lock(mutex_);
    bridge_connected_ = false;
    cv_.notify_all();
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

template <core::concepts::Game Game>
void WebPlayer<Game>::start_game() {
  action_ = -1;
  resign_ = false;

  if (first_game_) {
    launch_bridge();
    launch_frontend();

    std::cout << "Please open the frontend in your browser at:\n\n"
              << "    http://localhost:5173\n"
              << std::endl;

    {
      mit::unique_lock lock(mutex_);
      cv_.wait(lock, [this]() { return bridge_connected_; });
    }

    first_game_ = false;
  } else {
    mit::unique_lock lock(mutex_);
    cv_.wait(lock, [this]() { return ready_for_new_game_; });
    ready_for_new_game_ = false;
  }
  send_start_game();
}

template <core::concepts::Game Game>
void WebPlayer<Game>::receive_state_change(core::seat_index_t seat, const State& state,
                                           core::action_t action) {
  // TODO: I think getting mode from state is not right. This callback should really receive
  // the action_mode as a separate argument.
  send_state_update(seat, state, action, Game::Rules::get_action_mode(state));
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
  send_action_request(request.valid_actions);
  notification_unit_ = request.notification_unit;

  // NOTE: see long comment in GameServer.inl, in the next() method, about a race condition
  // consideration about this.
  return ActionResponse::yield();
}

template <core::concepts::Game Game>
void WebPlayer<Game>::end_game(const State& state, const ValueTensor& outcome) {
  boost::json::object msg;
  msg["type"] = "game_end";
  msg["payload"] = make_result_msg(state, outcome);
  send_msg(msg);
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_start_game_msg() {
  util::Rendering::Guard guard(util::Rendering::kText);

  State state;
  Game::Rules::init_state(state);

  boost::json::object payload;
  payload["board"] = IO::state_to_json(state);
  auto seat_assignments = boost::json::array();
  auto player_names = boost::json::array();
  for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
    seat_assignments.push_back(boost::json::value(IO::player_to_str(p)));
    player_names.push_back(boost::json::value(this->get_player_names()[p]));
  }
  payload["my_seat"] = IO::player_to_str(this->get_my_seat());
  payload["seat_assignments"] = seat_assignments;
  payload["player_names"] = player_names;
  return payload;
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_result_msg(const State& state,
                                                     const ValueTensor& outcome) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::object payload;
  constexpr int P = Game::Constants::kNumPlayers;

  auto array = Game::GameResults::to_value_array(outcome);

  char result_codes[P + 1];
  for (int p = 0; p < P; ++p) {
    if (array[p] == 1) {
      result_codes[p] = 'W';  // Win
    } else if (array[p] == 0) {
      result_codes[p] = 'L';  // Loss
    } else {
      result_codes[p] = 'D';  // Draw
    }
  }
  result_codes[P] = '\0';  // Null-terminate the string
  payload["result_codes"] = std::string(result_codes);

  return payload;
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_action_request_msg(const ActionMask& valid_actions) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::array legal_move_indices;
  for (int i : bitset_util::on_indices(valid_actions)) {
    legal_move_indices.push_back(i);
  }

  boost::json::object payload;
  payload["legal_moves"] = legal_move_indices;
  return payload;
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_state_update_msg(core::seat_index_t seat,
                                                           const State& state,
                                                           core::action_t last_action,
                                                           core::action_mode_t last_mode) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::object payload;
  payload["board"] = IO::state_to_json(state);
  payload["seat"] = IO::player_to_str(seat);
  payload["last_action"] = IO::action_to_str(last_action, last_mode);
  return payload;
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_msg(const boost::json::object& msg) {
  std::string out = boost::json::serialize(msg) + "\n";
  boost::asio::write(socket_, boost::asio::buffer(out));
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
    bridge_connected_ = true;
  }
  cv_.notify_all();

  while (bridge_connected_) {
    try {
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
        action_ = idx;
        notification_unit_.yield_manager->notify(notification_unit_);
      } else if (type == "new_game") {
        mit::unique_lock lock(mutex_);
        ready_for_new_game_ = true;
        lock.unlock();
        cv_.notify_all();
      } else if (type == "resign") {
        resign_ = true;
        notification_unit_.yield_manager->notify(notification_unit_);
      } else {
        throw util::Exception("Unknown message type: {}", type);
      }
    } catch (const std::exception& ex) {
      LOG_INFO("WebPlayer: connection closed or error: {}", ex.what());
      bridge_connected_ = false;
      cv_.notify_all();
      break;
    }
  }
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_start_game() {
  boost::json::object msg;
  msg["type"] = "start_game";
  msg["payload"] = make_start_game_msg();

  send_msg(msg);
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_state_update(core::seat_index_t seat, const State& state,
                                        core::action_t last_action, core::action_mode_t last_mode) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::object msg;
  msg["type"] = "state_update";
  msg["payload"] = make_state_update_msg(seat, state, last_action, last_mode);

  send_msg(msg);
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_action_request(const ActionMask& valid_actions) {
  boost::json::object msg;
  msg["type"] = "action_request";
  msg["payload"] = make_action_request_msg(valid_actions);

  send_msg(msg);
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
