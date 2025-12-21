#pragma once

#include "core/ActionResponse.hpp"
#include "games/tictactoe/Game.hpp"
#include "generic_players/HumanTuiPlayer.hpp"

namespace tictactoe {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 public:
  using base_t = generic::HumanTuiPlayer<Game>;
  using State = Game::State;

 private:
  core::ActionResponse prompt_for_action(const ActionRequest&) override;
  void print_state(const State&, bool terminal) override;
};

}  // namespace tictactoe

#include "inline/games/tictactoe/players/HumanTuiPlayer.inl"
