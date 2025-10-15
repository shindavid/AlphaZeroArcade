#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/VerboseManager.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <boost/asio.hpp>
#include <boost/json.hpp>
#include <boost/process.hpp>

namespace generic {

template <core::concepts::Game Game>
class WebPlayer : public core::AbstractPlayer<Game> {
 public:
  using GameClass = Game;
  using GameTypes = Game::Types;
  using base_t = core::AbstractPlayer<Game>;
  using IO = Game::IO;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using GameResultTensor = Game::Types::GameResultTensor;

  static_assert(core::concepts::WebGameIO<IO, GameTypes>, "IO must satisfy WebGameIO");

  WebPlayer();
  ~WebPlayer();

  bool start_game() override;
  void receive_state_change(core::seat_index_t, const State&, core::action_t) override;
  ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const State&, const GameResultTensor&) override;
  bool disable_progress_bar() const override { return true; }

 protected:
  // Optional: override this to provide a game-specific start_game message.
  // By default, it returns something like:
  //
  // {
  //   "board": IO::state_to_json(state),
  //   "my_seat": "X",
  //   "seat_assignments": ["X", "O"],
  //   "player_names": ["MCTS-C", "Human"],
  // }
  virtual boost::json::object make_start_game_msg();

  // Optional: override this to provide a game-specific result message.
  // By default, it returns something like:
  //
  // {
  //   "result_codes": "WL"
  // }
  //
  // The result code string is a sequence of characters, one for each player:
  // 'W' for win, 'L' for loss, 'D' for draw.
  //
  // In a game like go, we might specialize by adding the margin of victory.
  //
  // In a game like chess, we might specialize by adding detail about why the game ended (e.g.,
  // stalemate, threefold repetition, etc.).
  virtual boost::json::object make_result_msg(const State& state, const GameResultTensor& outcome);

  // Optional: override this to provide a game-specific action request msg.
  // By default, it returns something like:
  //
  // {
  //   "legal_moves": [0, 3, 4],
  // }
  //
  // The moves are by index.
  //
  // For games with more complex actions, we likely want to override this so that the frontend
  // does not need to know the action->index mapping.
  virtual boost::json::object make_action_request_msg(const ActionMask& valid_actions);

  // Construct json object that the frontend can use to display the state.
  //
  // By default, constructs a dict via:
  //
  // {
  //   "board": IO::state_to_json(state),
  //   "last_action": IO::action_to_str(last_action, last_mode),
  //   "seat": IO::player_to_str(seat)
  // }
  //
  // Can be overridden to add more fields, or change the representation.
  //
  // TODO: when playing against an MCTS player, this is where we should add MCTS stats for
  // visualization in the frontend. This too should have reasonable defaults built into this
  // base class.
  virtual boost::json::object make_state_update_msg(core::seat_index_t seat, const State& state,
                                                    core::action_t last_action,
                                                    core::action_mode_t last_mode);

 private:
  void send_start_game();
  void send_state_update(core::seat_index_t seat, const State& state,
                         core::action_t last_action = -1, core::action_mode_t last_mode = 0);
  void send_action_request(const ActionMask& valid_actions);

  void send_msg(const boost::json::object& msg);
  void launch_bridge();
  void launch_frontend();
  void response_loop();

  boost::asio::ip::tcp::acceptor create_acceptor();

  int engine_port_ = 48040;  // TODO: Make this configurable
  int bridge_port_ = 52528;  // TODO: Make this configurable
  int vite_port_ = 5173;     // TODO: Make this configurable

  boost::asio::io_context io_context_;
  boost::asio::ip::tcp::acceptor acceptor_;
  boost::asio::ip::tcp::socket socket_;
  mit::thread thread_;
  mit::condition_variable cv_;
  mit::mutex mutex_;

  boost::process::child* bridge_process_ = nullptr;
  boost::process::child* frontend_process_ = nullptr;

  core::YieldNotificationUnit notification_unit_;
  core::action_t action_ = -1;
  bool first_game_ = true;
  bool resign_ = false;
  bool bridge_connected_ = false;
  bool ready_for_new_game_ = false;
};

}  // namespace generic

#include "inline/generic_players/WebPlayer.inl"
