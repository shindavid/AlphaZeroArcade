#pragma once

#include <core/BasicTypes.hpp>
#include <generic_players/HumanTuiPlayer.hpp>
#include <games/chess/Game.hpp>

namespace chess {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 private:
  using State = Game::State;
  core::action_t prompt_for_action(const State&, const ActionMask&) override;
};

}  // namespace chess

#include <inline/games/chess/players/HumanTuiPlayer.inl>
