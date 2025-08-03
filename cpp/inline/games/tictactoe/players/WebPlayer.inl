#include "games/tictactoe/players/WebPlayer.hpp"

#include "util/Exceptions.hpp"
#include "util/LoggingUtil.hpp"
#include "util/mit/mit.hpp"

#include <boost/process.hpp>

namespace tictactoe {

inline WebPlayer::WebPlayer() : acceptor_(create_acceptor()), socket_(io_context_) {
  thread_ = mit::thread([this]() { response_loop(); });
}

inline WebPlayer::~WebPlayer() {
  if (bridge_process_) {
    bridge_process_->terminate();
    delete bridge_process_;
  }
  if (frontend_process_) {
    frontend_process_->terminate();
    delete frontend_process_;
  }
}

inline void WebPlayer::start_game() {
  if (first_game_) {
    // sleep for one second to ensure response_loop() is ready
    std::this_thread::sleep_for(std::chrono::seconds(1));
    launch_bridge();
    launch_frontend();

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

inline void WebPlayer::receive_state_change(core::seat_index_t seat, const State& state,
                                            core::action_t action) {
  send_state(state);
}

inline WebPlayer::ActionResponse WebPlayer::get_action_response(const ActionRequest& request) {
  if (action_ != -1) {
    core::action_t action = action_;
    action_ = -1;
    return action;
  }
  send_state(request.state, this->get_my_seat());
  notification_unit_ = request.notification_unit;

  // NOTE: see long comment in GameServer.inl, in the next() method, about a race condition
  // consideration about this.
  mit::unique_lock lock(mutex_);
  ready_for_response_ = true;
  lock.unlock();
  cv_.notify_all();

  return ActionResponse::yield();
}

inline void WebPlayer::end_game(const State& state, const ValueTensor& outcome) {
  auto array = Game::GameResults::to_value_array(outcome);

  core::seat_index_t winner = -1;
  for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
    if (array[p] == 1) {
      winner = p;
      break;
    }
  }

  boost::json::object msg;
  msg["type"] = "game_end";
  boost::json::object payload;
  if (winner == -1) {
    payload["result"] = "draw";
  } else {
    payload["result"] = "win";
    payload["winner"] = IO::player_to_str(winner);
  }
  msg["payload"] = payload;
  std::string out = boost::json::serialize(msg) + "\n";
  RELEASE_ASSERT(connected_);
  boost::asio::write(socket_, boost::asio::buffer(out));
}

inline void WebPlayer::launch_bridge() {
  namespace bp = boost::process;

  bp::environment env = boost::this_process::environment();
  env["BRIDGE_PORT"] = std::to_string(bridge_port_);
  env["ENGINE_PORT"] = std::to_string(engine_port_);
  env["SPAWN_ENGINE"] = "false";

  bridge_process_ = new bp::child("npm start", bp::start_dir = "/workspace/repo/web/server",
                                  bp::std_out > "/workspace/repo/bridge2.log",
                                  bp::std_err > "/workspace/repo/bridge2.log", env);

  LOG_INFO("Web player launched bridge process on port {}", bridge_port_);
}

inline void WebPlayer::launch_frontend() {
  namespace bp = boost::process;

  bp::environment env = boost::this_process::environment();
  env["VITE_BRIDGE_PORT"] = std::to_string(bridge_port_);

  frontend_process_ =
    new bp::child("npm run dev", bp::start_dir = "/workspace/repo/web/games/tictactoe",
                  bp::std_out > "/workspace/repo/frontend2log",
                  bp::std_err > "/workspace/repo/frontend2.log", env);

  LOG_INFO("Web player launched frontend process");
}

inline void WebPlayer::response_loop() {
  LOG_INFO("Web player is waiting for a connection on port {}", engine_port_);
  acceptor_.accept(socket_);
  LOG_INFO("Web player connected to client.");

  {
    mit::unique_lock lock(mutex_);
    connected_ = true;
  }

  // TODO: wire end_session() to WebPlayer, and use that to break this loop
  // And/or: rely on socket_.is_open()
  while (true) {
    {
      mit::unique_lock lock(mutex_);

      if (pending_send_) {
        write_to_socket(pending_state_, pending_seat_);
        pending_send_ = false;
      }

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
      notification_unit_.yield_manager->notify(notification_unit_);
    } else if (type == "resign") {
      throw util::CleanException("TODO: implement resignation");
    } else {
      throw util::Exception("Unknown message type: {}", type);
    }
  }
}

// TODO: we cannot send anything to socket_ until after acceptor_.accept() is called in
// response_loop(). If we hit this method before that point, we need to store the state, and send it
// later.
inline void WebPlayer::send_state(const State& state, core::seat_index_t seat) {
  mit::unique_lock lock(mutex_);
  if (!connected_) {
    pending_send_ = true;
    pending_state_ = state;
    pending_seat_ = seat;
    return;
  }
  write_to_socket(state, seat);
}

inline void WebPlayer::write_to_socket(const State& state, core::seat_index_t seat) {
  boost::json::object msg;
  msg["type"] = "state_update";
  boost::json::object payload;
  payload["board"] = IO::compact_state_repr(state);
  if (seat >= 0) {
    payload["turn"] = IO::player_to_str(seat);
  }
  msg["payload"] = payload;
  std::string out = boost::json::serialize(msg) + "\n";
  boost::asio::write(socket_, boost::asio::buffer(out));
}


inline boost::asio::ip::tcp::acceptor WebPlayer::create_acceptor() {
  boost::asio::ip::tcp::acceptor acceptor(io_context_);
  acceptor.open(boost::asio::ip::tcp::v4());
  acceptor.set_option(boost::asio::socket_base::reuse_address(true));
  acceptor.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), engine_port_));
  acceptor.listen();
  return acceptor;
}

}  // namespace tictactoe
