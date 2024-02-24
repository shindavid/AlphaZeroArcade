#pragma once

#include <generic_players/HumanTuiPlayer.hpp>
#include <othello/GameState.hpp>

namespace othello {

class HumanTuiPlayer : public generic::HumanTuiPlayer<GameState> {
 private:
  Action prompt_for_action(const GameState&, const ActionMask&) override;
  int prompt_for_action_helper(const GameState&, const ActionMask&);
};

}  // namespace othello

#include <inline/othello/players/HumanTuiPlayer.inl>
