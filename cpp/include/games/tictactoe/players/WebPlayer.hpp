#pragma once

#include "core/AbstractPlayer.hpp"
#include "games/tictactoe/Game.hpp"
#include "util/mit/mit.hpp"

#include <boost/asio.hpp>

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

  void start_game() override;
  void receive_state_change(core::seat_index_t, const State&, core::action_t) override;
  ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const State&, const ValueTensor&) override;

 private:
  void response_loop();
  void send_state(const State& state, core::seat_index_t seat=-1);
  boost::asio::ip::tcp::acceptor create_acceptor();

  boost::asio::io_context io_context_;
  boost::asio::ip::tcp::acceptor acceptor_;
  boost::asio::ip::tcp::socket socket_;
  mit::thread thread_;
  mit::condition_variable cv_;
  mit::mutex mutex_;
  int port_ = 4000;  // TODO: Make this configurable

  core::YieldNotificationUnit notification_unit_;
  core::action_t action_ = -1;
  bool first_game_ = true;
  bool ready_for_response_ = false;
};

}  // namespace tictactoe

#include "inline/games/tictactoe/players/WebPlayer.inl"
