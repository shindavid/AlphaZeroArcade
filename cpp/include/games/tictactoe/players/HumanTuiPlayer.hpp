#pragma once

#include <generic_players/HumanTuiPlayer.hpp>
#include <games/tictactoe/GameState.hpp>
#include <games/tictactoe/players/PerfectPlayer.hpp>

namespace tictactoe {

class HumanTuiPlayer : public generic::HumanTuiPlayer<GameState> {
 public:
  using base_t = generic::HumanTuiPlayer<GameState>;

 private:
  Action prompt_for_action(const GameState&, const ActionMask&) override;
  int prompt_for_action_helper(const GameState&, const ActionMask&);
  void print_state(const GameState&, bool terminal) override;
};

}  // namespace tictactoe

#include <inline/games/tictactoe/players/HumanTuiPlayer.inl>
