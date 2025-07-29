#pragma once

#include "core/AbstractPlayer.hpp"
#include "games/tictactoe/Game.hpp"

// TODO: Pull out a generic::WebPlayer base-class that can be used for any game, and have
// tictactoe::WebPlayer inherit from it.

namespace tictactoe {

class WebPlayer : public core::AbstractPlayer<Game> {
 public:
  using base_t = core::AbstractPlayer<Game>;
  using IO = Game::IO;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using ValueTensor = Game::Types::ValueTensor;

  void start_game() override;
  void receive_state_change(core::seat_index_t, const State&, core::action_t) override;
  ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const State&, const ValueTensor&) override;
};

}  // namespace tictactoe

#include "inline/games/tictactoe/players/WebPlayer.inl"
