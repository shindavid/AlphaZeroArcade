#pragma once

#include <common/players/HumanTuiPlayer.hpp>
#include <games/othello/GameState.hpp>

namespace othello {

class HumanTuiPlayer : public common::HumanTuiPlayer<GameState> {
 private:
  Action prompt_for_action(const GameState&, const ActionMask&) override;
  int prompt_for_action_helper(const GameState&, const ActionMask&);
};

}  // namespace othello

#include <games/othello/players/inl/HumanTuiPlayer.inl>
