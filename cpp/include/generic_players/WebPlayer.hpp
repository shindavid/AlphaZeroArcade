#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/BacktrackUpdate.hpp"
#include "core/BasicTypes.hpp"
#include "core/StateChangeUpdate.hpp"
#include "core/WebManager.hpp"
#include "core/WebManagerClient.hpp"
#include "core/concepts/GameConcept.hpp"

#include <boost/json/object.hpp>

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
  using ActionRequest = core::ActionRequest<Game>;
  using GameResultTensor = Game::Types::GameResultTensor;
  using ActionMask = Game::Types::ActionMask;
  using StateChangeUpdate = core::StateChangeUpdate<Game>;
  using BacktrackUpdate = core::BacktrackUpdate<Game>;

  WebPlayer() : WebManagerClient(std::in_place_type<WebManager>) {}
  virtual ~WebPlayer() = default;

  // AbstractPlayer interface
  bool start_game() override;
  void receive_state_change(const StateChangeUpdate&) override;
  core::ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const State&, const GameResultTensor&) override;
  bool disable_progress_bar() const override { return true; }

  // WebManagerClient interface
  void handle_action(const boost::json::object& payload, core::seat_index_t seat) override;
  void handle_resign(core::seat_index_t seat) override;
  void handle_backtrack(core::game_tree_index_t index, core::seat_index_t seat) override;

 protected:
  core::ActionResponse get_web_response(const ActionRequest& request,
                                        const core::ActionResponse& proposed_response);
  void initialize_game();
  void send_state_update(const StateChangeUpdate&);
  void send_result_msg(const State& state, const GameResultTensor& outcome);

 private:
  /*
   * Payload encapsulates a single payload to be sent to the web frontend. It looks like:
   * {
   *   "type": "start_game" | "action_request" | "state_update" | "game_end" | "tree_node",
   *   "cache_key": "<type>:<index>",
   *   ... other fields depending on type ...
   * }
   */
  class Payload {
   public:
    enum Type { kStartGame, kActionRequest, kStateUpdate, kGameEnd, kTreeNode };
    Payload(Type t, int cache_key_index = -1) : type_(t), cache_key_index_(cache_key_index) {};

    boost::json::object to_json() const;
    template <typename T>
    void add_field(const std::string& key, T&& value);

   private:
    Type type_;
    int cache_key_index_;
    boost::json::object obj_;
  };
  /*
   * Message encapsulates a message to be sent to the web frontend. It can contain multiple
   * payloads. It looks like:
   * {
   *   "bridge_action": "reset" | "update",
   *   "payloads": [
   *     {"type": "start_game", ...},
   *     {"type": "state_update", ...},
   *     ...
   *   ]
   */
  class Message {
   public:
    enum BridgeAction { kReset, kUpdate };
    Message(BridgeAction bridge_action);
    void send();

    void add_payload(boost::json::object&& payload) {
      msg_["payloads"].as_array().push_back(std::move(payload));
    }

    void add_payload(const boost::json::object& payload) {
      msg_["payloads"].as_array().push_back(payload);
    }

   private:
    boost::json::object msg_;
  };

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

  // Optional: override this to provide a tree update message.
  // By default, returns a dict like:
  //
  // {
  //   "index": update.index,
  //   "parent_index": update.parent_index,
  //   "seat": std::string(1, Game::IO::kSeatChars[update.seat])
  // }

  virtual boost::json::object make_tree_node_msg(const StateChangeUpdate&);

 private:
  core::YieldNotificationUnit notification_unit_;
  core::action_t action_ = -1;
  bool resign_ = false;
  core::game_tree_index_t backtrack_index_ = -1;
};

}  // namespace generic

#include "inline/generic_players/WebPlayer.inl"
