#pragma once

#include <common/players/HumanTuiPlayer.hpp>
#include <games/othello/GameState.hpp>

namespace othello {

class HumanTuiPlayer : public common::HumanTuiPlayer<GameState> {
private:
  core::action_index_t prompt_for_action(const GameState&, const ActionMask&) override;
};

}  // namespace othello

#include <games/othello/players/inl/HumanTuiPlayer.inl>
