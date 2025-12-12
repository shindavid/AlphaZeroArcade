#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "generic_players/WebPlayer.hpp"

namespace generic {

/*
 * AnalysisPlayer wraps an AbstractPlayer and adds web-based communication to enable interactive
 * analysis. It uses WebManager to interface with a web frontend, allowing users to inspect the
 * wrapped player's reasoning and optionally override its moves. AnalysisPlayer registers handlers
 * for incoming messages such as "make_move" and "resign", supporting interactive gameplay with
 * detailed insight into the player's decisions.
 */
template <core::concepts::Game Game>
class AnalysisPlayer : public WebPlayer<Game> {
 public:
  using State = Game::State;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using GameResultTensor = Game::Types::GameResultTensor;
  using ActionMask = Game::Types::ActionMask;

  AnalysisPlayer(core::AbstractPlayer<Game>* wrapped_player) : wrapped_player_(wrapped_player) {}
  ~AnalysisPlayer() { delete wrapped_player_; }

  // TODO: we should probably have start_game() use the kContinue/kYield framework,
  // so that GameServer isn 't blocked indefinitely here. This doesn' t matter right now,
  // but if in the future we have a single GameServer hosting multiple simultaneous web games,
  // this can cause one AFK player blocking all games.
  bool start_game() override;
  ActionResponse get_action_response(const ActionRequest&, const void*&) override;
  void receive_state_change(core::seat_index_t seat, const State& state,
                            core::action_t action) override;
  void end_game(const State& state, const GameResultTensor& outcome) override;

 private:
  core::AbstractPlayer<Game>* const wrapped_player_;
};

}  // namespace generic

#include "inline/generic_players/AnalysisPlayer.inl"
