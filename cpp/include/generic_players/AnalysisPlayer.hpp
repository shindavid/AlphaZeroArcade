#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace generic {

template <core::concepts::Game Game>
class AnalysisPlayer : public core::AbstractPlayer<Game> {
 public:
  using State = typename Game::State;
  using ActionRequest = typename Game::Types::ActionRequest;
  using ActionResponse = typename Game::Types::ActionResponse;
  using GameResultTensor = typename Game::Types::GameResultTensor;
  using ActionMask = typename Game::Types::ActionMask;

  AnalysisPlayer(core::AbstractPlayer<Game>* wrapped_player);
  ~AnalysisPlayer() { delete wrapped_player_; }

  bool start_game() override;
  ActionResponse get_action_response(const ActionRequest& request) override;

  void receive_state_change(core::seat_index_t seat, const State& state,
                            core::action_t action) override {
    wrapped_player_->receive_state_change(seat, state, action);
  }

  void end_game(const State& state, const GameResultTensor& outcome) override {
    wrapped_player_->end_game(state, outcome);
  }

  bool disable_progress_bar() const override { return true; }

  void set_action(const boost::json::object& payload);
  void set_resign(const boost::json::object& payload);

 private:
  boost::json::object make_start_game_msg();
  void send_action_request(const ActionMask& valid_actions, core::action_t proposed_action);
  boost::json::object make_action_request_msg(const ActionMask& valid_actions,
                                              core::action_t proposed_action);

  core::AbstractPlayer<Game>* wrapped_player_;
  mit::mutex mutex_;
  mit::condition_variable cv_;
  core::YieldNotificationUnit notification_unit_;
  core::action_t action_ = -1;
  bool first_game_ = true;
  bool resign_ = false;
  bool ready_for_new_game_ = false;
};

}  // namespace generic

#include "inline/generic_players/AnalysisPlayer.inl"
