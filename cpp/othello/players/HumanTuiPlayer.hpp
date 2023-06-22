#pragma once

#include <common/HumanTuiPlayer.hpp>
#include <othello/GameState.hpp>

namespace othello {

class HumanTuiPlayer : public common::HumanTuiPlayer<GameState> {
private:
  core::action_index_t prompt_for_action(const GameState&, const ActionMask&) override;
};

}  // namespace othello

#include <othello/players/inl/HumanTuiPlayer.inl>
