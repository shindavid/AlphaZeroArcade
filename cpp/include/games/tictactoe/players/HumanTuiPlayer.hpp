#pragma once

#include <core/BasicTypes.hpp>
#include <generic_players/HumanTuiPlayer.hpp>
#include <games/tictactoe/Game.hpp>
#include <games/tictactoe/players/PerfectPlayer.hpp>

namespace tictactoe {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 public:
  using base_t = generic::HumanTuiPlayer<Game>;
  using FullState = Game::FullState;

 private:
  core::action_t prompt_for_action(const FullState&, const ActionMask&) override;
  void print_state(const FullState&, bool terminal) override;
};

}  // namespace tictactoe

#include <inline/games/tictactoe/players/HumanTuiPlayer.inl>
