#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "generic_players/WebPlayerBase.hpp"

namespace generic {

/*
 * AnalysisPlayer wraps an AbstractPlayer and adds web-based communication to enable interactive
 * analysis. It uses WebManager to interface with a web frontend, allowing users to inspect the
 * wrapped player's reasoning and optionally override its moves. AnalysisPlayer registers handlers
 * for incoming messages such as "make_move" and "resign", supporting interactive gameplay with
 * detailed insight into the player's decisions.
 */
template <core::concepts::Game Game>
class AnalysisPlayer : public WebPlayerBase<Game>{
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
                            core::action_t action) override;
  void end_game(const State& state, const GameResultTensor& outcome) override;
  bool disable_progress_bar() const override { return true; }

 private:
  core::AbstractPlayer<Game>* wrapped_player_;

};

}  // namespace generic

#include "inline/generic_players/AnalysisPlayer.inl"
