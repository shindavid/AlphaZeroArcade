#pragma once

#include <generic_players/HumanTuiPlayer.hpp>
#include <tictactoe/GameState.hpp>
#include <tictactoe/players/PerfectPlayer.hpp>

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

#include <inline/tictactoe/players/HumanTuiPlayer.inl>
