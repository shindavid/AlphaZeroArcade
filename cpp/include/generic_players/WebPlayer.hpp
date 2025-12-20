#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/WebManager.hpp"
#include "core/WebManagerClient.hpp"
#include "core/concepts/GameConcept.hpp"

namespace generic {
/*
 * WebPlayer provides core functionality for web-based players, handling communication with
 * the web frontend via WebManager. It implements the AbstractPlayer interface and the
 * WebManagerClient interface to facilitate interactive gameplay through a web interface.
 */
template <core::concepts::Game Game>
class WebPlayer : public core::WebManagerClient, public core::AbstractPlayer<Game> {
 public:
  using GameClass = Game;
  using WebManager = core::WebManager<Game>;
  using State = Game::State;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using GameResultTensor = Game::Types::GameResultTensor;
  using ActionMask = Game::Types::ActionMask;
  using StateChangeUpdate = Game::Types::StateChangeUpdate;

  WebPlayer() : WebManagerClient(std::in_place_type<WebManager>) {}
  virtual ~WebPlayer() = default;

  // AbstractPlayer interface
  bool start_game() override;
  void receive_state_change(const StateChangeUpdate&) override;
  ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const State&, const GameResultTensor&) override;
  bool disable_progress_bar() const override { return true; }

  // WebManagerClient interface
  void handle_action(const boost::json::object& payload, core::seat_index_t seat) override;
  void handle_resign(core::seat_index_t seat) override;

 protected:
  ActionResponse get_web_response(const ActionRequest& request,
                                  const ActionResponse& proposed_response);
  void initialize_game();
  void send_state_update(const StateChangeUpdate&);
  void send_result_msg(const State& state, const GameResultTensor& outcome);

 private:
  void send_start_game();
  void send_action_request(const ActionMask& valid_actions, core::action_t proposed_action);

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

  // Optional: override this to provide a game-specific action request msg.
  // By default, it returns something like:
  //
  // {
  //   "legal_moves": [0, 3, 4],
  //   "seat": 0,
  //   "proposed_action": 4,
  //   "verbose_info": { ... }
  // }
  //
  // The moves are by index.
  //
  // For games with more complex actions, we likely want to override this so that the frontend
  // does not need to know the action->index mapping.
  virtual boost::json::object make_action_request_msg(const ActionMask& valid_actions,
                                                      core::action_t proposed_action);

  // Construct json object that the frontend can use to display the state.
  //
  // By default, constructs a dict via:
  //
  // {
  //   "board": IO::state_to_json(state),
  //   "seat": seat,
  //   "last_action": IO::action_to_str(last_action, last_mode),
  // }
  //
  // Can be overridden to add more fields, or change the representation.
  virtual boost::json::object make_state_update_msg(const StateChangeUpdate&);

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

 private:
  core::YieldNotificationUnit notification_unit_;
  core::action_t action_ = -1;
  bool resign_ = false;
};

}  // namespace generic

#include "inline/generic_players/WebPlayer.inl"
