#pragma once

#include <common/players/HumanTuiPlayer.hpp>
#include <games/tictactoe/GameState.hpp>
#include <games/tictactoe/players/PerfectPlayer.hpp>

namespace tictactoe {

class HumanTuiPlayer : public common::HumanTuiPlayer<GameState> {
 public:
  using base_t = common::HumanTuiPlayer<GameState>;

 private:
  Action prompt_for_action(const GameState&, const ActionMask&) override;
  int prompt_for_action_helper(const GameState&, const ActionMask&);
  void print_state(const GameState&, bool terminal) override;
};

}  // namespace tictactoe

#include <games/tictactoe/players/inl/HumanTuiPlayer.inl>
