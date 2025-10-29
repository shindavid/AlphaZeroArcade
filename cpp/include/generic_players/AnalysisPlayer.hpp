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

  AnalysisPlayer(core::AbstractPlayer<Game>* wrapped_player) : wrapped_player_(wrapped_player) {}
  ~AnalysisPlayer() { delete wrapped_player_; }

  std::string get_name() const {
    return "AnalysisPlayer(" + wrapped_player_->get_name() + ")";
  }

  bool start_game() override;
  void receive_state_change(core::seat_index_t seat, const State& state,
                            core::action_t action) override {
    wrapped_player_->receive_state_change(seat, state, action);
  }
  ActionResponse get_action_response(const ActionRequest& request) override {
    return wrapped_player_->get_action_response(request);
  }
  void end_game(const State& state, const GameResultTensor& outcome) override {
    wrapped_player_->end_game(state, outcome);
  }

  bool disable_progress_bar() const override { return true; }

 private:
  void send_start_game();
  boost::json::object make_start_game_msg();

  core::AbstractPlayer<Game>* wrapped_player_;
  mit::mutex mutex_;
  mit::condition_variable cv_;
  core::action_t action_ = -1;
  bool first_game_ = true;
  bool resign_ = false;
  bool ready_for_new_game_ = false;
};

}  // namespace generic

#include "inline/generic_players/AnalysisPlayer.inl"
