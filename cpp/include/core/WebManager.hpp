#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <array>
#include <boost/asio.hpp>
#include <boost/json.hpp>
#include <boost/process.hpp>
#include <functional>
#include <unordered_map>

namespace core {

using MsgType = std::string;
using HandlerFunc = std::function<void(const boost::json::object& payload)>;
using HandlerFuncMap = std::unordered_map<MsgType, HandlerFunc>;

/*
 * WebManager is a singleton responsible for communication between the game server and the web-based
 * frontend through a bridge process. It launches both the bridge and frontend upon construction and
 * processes incoming bridge messages on a dedicated thread. Player instances (e.g., AnalysisPlayer,
 * WebPlayer) can register handlers for specific message types, which are automatically invoked when
 * messages of those types are received.
 */
template<concepts::Game Game>
struct WebManager {
  using Handlers = std::array<HandlerFuncMap, Game::Constants::kNumPlayers>; // idx by seat

  WebManager();
  ~WebManager();
  static WebManager* get_instance();
  void wait_for_connection();
  void wait_for_new_game_ready();

  /*
   * A starter is a player that initiates a new game in the front end. The starter sends the initial
   * game state. Only one player performs the starter role for each game. At the end of each game,
   * the starter role is cleared, allowing another player to become the starter for the next game.
   */
  bool become_starter();
  void clear_starter();

  void register_handler(seat_index_t seat, HandlerFuncMap&& handler_map) {
    handlers_[seat] = std::move(handler_map);
  }

  /*
   * All handlers are cleared at the end of each game to prevent retaining stale handlers from
   * previous sessions. Although new handlers are typically registered at the start of every game,
   * this serves as a defensive safeguard.
   */
  void clear_handlers();
  void send_msg(const boost::json::object& msg);

 private:
  boost::asio::ip::tcp::acceptor create_acceptor();

  void launch_bridge();
  void launch_frontend();
  void response_loop();

  int engine_port_ = 48040;  // TODO: Make this configurable
  int bridge_port_ = 52528;  // TODO: Make this configurable
  int vite_port_ = 5173;     // TODO: Make this configurable

  boost::asio::io_context io_context_;
  boost::asio::ip::tcp::acceptor acceptor_;
  boost::asio::ip::tcp::socket socket_;
  mit::thread thread_;
  mit::mutex mutex_;
  mit::condition_variable cv_;

  bool bridge_connected_ = false;
  bool ready_for_new_game_ = true;
  bool has_starter_ = false;

  boost::process::child* bridge_process_ = nullptr;
  boost::process::child* frontend_process_ = nullptr;

  Handlers handlers_;
};

}  // namespace core

#include "inline/core/WebManager.inl"
