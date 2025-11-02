#pragma once

#include "boost/json/object.hpp"
#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace generic {
/*
 * WebPlayerBase provides core functionality for web-based players, handling communication with
 * the web frontend via WebManager. It includes methods to send game start messages, action
 * requests, state updates, and game results to the frontend. Derived classes can override message
 * construction methods to customize the payloads sent to the web interface.
 *
 * Note: WebPlayerBase does not have the necessary methods to function as a player on its own. It is
 * intended to be subclassed by specific player implementations (e.g., WebPlayer, AnalysisPlayer)
 * that provide the required player interface.
 */
template <core::concepts::Game Game>
class WebPlayerBase : public core::AbstractPlayer<Game> {
 public:
  using State = typename Game::State;
  using ActionRequest = typename Game::Types::ActionRequest;
  using ActionResponse = typename Game::Types::ActionResponse;
  using GameResultTensor = typename Game::Types::GameResultTensor;
  using ActionMask = typename Game::Types::ActionMask;

  void set_action(const boost::json::object& payload);
  void set_resign(const boost::json::object& payload);

 protected:
  void send_start_game();
  void send_action_request(const ActionMask& valid_actions, core::action_t proposed_action);
  void send_state_update(core::seat_index_t seat, const State& state, core::action_t last_action,
                         core::action_mode_t last_mode);

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
  //
  // TODO: when playing against an MCTS player, this is where we should add MCTS stats for
  // visualization in the frontend. This too should have reasonable defaults built into this
  // base class.
  virtual boost::json::object make_state_update_msg(core::seat_index_t seat, const State& state,
                                                    core::action_t last_action,
                                                    core::action_mode_t last_mode);

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

  core::YieldNotificationUnit notification_unit_;
  core::action_t action_ = -1;
  bool resign_ = false;
};

}  // namespace generic

#include "inline/generic_players/WebPlayerBase.inl"