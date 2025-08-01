#include "games/tictactoe/players/WebPlayer.hpp"

#include "util/Exceptions.hpp"
#include "util/LoggingUtil.hpp"
#include "util/mit/mit.hpp"

namespace tictactoe {

inline WebPlayer::WebPlayer() : acceptor_(create_acceptor()), socket_(io_context_) {
  thread_ = mit::thread([this]() { response_loop(); });
}

inline void WebPlayer::start_game() {
  if (first_game_) {
    LOG_INFO("Web player is waiting for a connection on port {}", port_);
    acceptor_.accept(socket_);
    LOG_INFO("Web player connected to client.");

    first_game_ = false;
    return;
  }

  // Wait for player to click "Start Game" in the frontend
  while (true) {
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
  boost::asio::write(socket_, boost::asio::buffer(out));
}

inline void WebPlayer::response_loop() {
  // TODO: wire end_session() to WebPlayer, and use that to break this loop
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

      // TODO: extend Game::IO interface to have a str_to_action() method, and use that instead of
      // doing (idx - 1) here.
      action_ = idx - 1;  // Convert to zero-based index
      notification_unit_.yield_manager->notify(notification_unit_);
    } else if (type == "resign") {
      throw util::CleanException("TODO: implement resignation");
    } else {
      throw util::Exception("Unknown message type: {}", type);
    }
  }
}

inline void WebPlayer::send_state(const State& state, core::seat_index_t seat) {
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
  acceptor.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port_));
  acceptor.listen();
  return acceptor;
}

}  // namespace tictactoe
