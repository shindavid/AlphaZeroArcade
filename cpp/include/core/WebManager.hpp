#pragma once

#include "core/concepts/GameConcept.hpp"
#include "util/OsUtil.hpp"

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

template<concepts::Game Game>
struct WebManager {
  using Handlers = std::array<HandlerFuncMap, Game::Constants::kNumPlayers>; // idx by seat

  WebManager();
  ~WebManager();
  static WebManager* get_instance();
  void wait_for_connection();
  void wait_for_new_game_ready();

  void register_handler(seat_index_t seat, HandlerFuncMap&& handler_map) {
    handlers_[seat] = std::move(handler_map);
  }
  void send_msg(const boost::json::object& msg);
  bool become_starter();
  void clear_starter();

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
