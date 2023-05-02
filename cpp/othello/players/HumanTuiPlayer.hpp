#pragma once

#include <common/players/HumanTuiPlayer.hpp>
#include <othello/GameState.hpp>

namespace othello {

class HumanTuiPlayer : public common::HumanTuiPlayer<GameState> {
private:
  common::action_index_t prompt_for_action(const GameState&, const ActionMask&) override;
};

}  // namespace othello

#include <othello/players/inl/HumanTuiPlayer.inl>
