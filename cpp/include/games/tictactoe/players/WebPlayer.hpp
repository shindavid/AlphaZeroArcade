#pragma once

#include "core/AbstractPlayer.hpp"
#include "games/tictactoe/Game.hpp"
#include "util/mit/mit.hpp"

#include <boost/asio.hpp>
#include <boost/process.hpp>

// TODO: Pull out a generic::WebPlayer base-class that can be used for any game, and have
// tictactoe::WebPlayer inherit from it.

namespace tictactoe {

class WebPlayer : public core::AbstractPlayer<Game> {
 public:
  using base_t = core::AbstractPlayer<Game>;
  using IO = Game::IO;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using ValueTensor = Game::Types::ValueTensor;

  WebPlayer();
  ~WebPlayer();

  void start_game() override;
  void receive_state_change(core::seat_index_t, const State&, core::action_t) override;
  ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const State&, const ValueTensor&) override;
  bool disable_progress_bar() const override { return true; }

 private:
  void launch_bridge();
  void launch_frontend();
  void response_loop();
  void send_state(const State& state, core::seat_index_t seat = -1);
  void write_to_socket(const State& state, core::seat_index_t seat);
  boost::asio::ip::tcp::acceptor create_acceptor();

  int engine_port_ = 48040;  // TODO: Make this configurable
  int bridge_port_ = 52528;  // TODO: Make this configurable

  boost::asio::io_context io_context_;
  boost::asio::ip::tcp::acceptor acceptor_;
  boost::asio::ip::tcp::socket socket_;
  mit::thread thread_;
  mit::condition_variable cv_;
  mit::mutex mutex_;

  boost::process::child* bridge_process_ = nullptr;
  boost::process::child* frontend_process_ = nullptr;

  core::YieldNotificationUnit notification_unit_;
  State pending_state_;
  core::action_t action_ = -1;
  core::seat_index_t pending_seat_;
  bool pending_send_ = false;
  bool first_game_ = true;
  bool ready_for_response_ = false;
  bool resign_ = false;
  bool connected_ = false;
};

}  // namespace tictactoe

#include "inline/games/tictactoe/players/WebPlayer.inl"
