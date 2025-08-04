#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/concepts/Game.hpp"
#include "util/mit/mit.hpp"

#include <boost/asio.hpp>
#include <boost/process.hpp>

namespace generic {

template <core::concepts::Game Game>
class WebPlayer : public core::AbstractPlayer<Game> {
 public:
  using GameClass = Game;
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

 protected:
  // Optional: override this to provide a game-specific result message.
  // By default, it returns a string like "Player X wins!" or "Draw!"
  // In a game like go, it could be specialized to return "Black wins by 5.5 points".
  // In a game like chess, it could return "Draw due to threefold repetition".
  virtual std::string make_result_msg(const State& state, const ValueTensor& outcome);

  // Construct json object that the frontend can use to display the state.
  //
  // By default, constructs a dict via:
  //
  // {
  //   "board": IO::compact_state_repr(state),
  //   "last_action": IO::action_to_str(last_action, last_mode),
  //   "turn": IO::player_to_str(seat)
  // }
  //
  // Can be overridden to add more fields, or change the representation.
  //
  // TODO: when playing against an MCTS player, this is where we should add MCTS stats for
  // visualization in the frontend. This too should have reasonable defaults built into this
  // base class.
  virtual boost::json::object make_state_msg(const State& state, core::action_t last_action,
                                             core::action_mode_t last_mode);

 private:
  void send_state(const State& state, core::action_t last_action, core::action_mode_t last_mode);

  void launch_bridge();
  void launch_frontend();
  void response_loop();
  void write_to_socket(const State& state, core::action_t last_action,
                       core::action_mode_t last_mode);

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
  core::action_t pending_last_action_ = -1;
  core::action_t last_action_ = -1;
  core::action_mode_t pending_last_mode_ = 0;
  core::action_mode_t last_mode_ = 0;
  core::action_t action_ = -1;
  bool pending_send_ = false;
  bool first_game_ = true;
  bool ready_for_response_ = false;
  bool resign_ = false;
  bool connected_ = false;
};

}  // namespace generic

#include "inline/generic_players/WebPlayer.inl"
