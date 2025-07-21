#pragma once

#include "core/BasicTypes.hpp"
#include "games/tictactoe/Game.hpp"
#include "games/tictactoe/players/PerfectPlayer.hpp"
#include "generic_players/HumanTuiPlayer.hpp"

namespace tictactoe {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 public:
  using base_t = generic::HumanTuiPlayer<Game>;
  using State = Game::State;

 private:
  core::action_t prompt_for_action(const State&, const ActionMask&) override;
  void print_state(const State&, bool terminal) override;
};

}  // namespace tictactoe

#include "inline/games/tictactoe/players/HumanTuiPlayer.inl"
