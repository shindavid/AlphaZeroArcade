#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace generic {

template <core::concepts::Game Game>
class WebPlayerBase: public core::AbstractPlayer<Game> {
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
  virtual boost::json::object make_start_game_msg();
  virtual boost::json::object make_action_request_msg(const ActionMask& valid_actions,
                                              core::action_t proposed_action);
  virtual boost::json::object make_state_update_msg(core::seat_index_t seat, const State& state,
                                            core::action_t last_action,
                                            core::action_mode_t last_mode);
  virtual boost::json::object make_result_msg(const State& state, const GameResultTensor& outcome);

  core::YieldNotificationUnit notification_unit_;
  core::action_t action_ = -1;
  bool resign_ = false;
};

}  // namespace generic

#include "inline/generic_players/WebPlayerBase.inl"