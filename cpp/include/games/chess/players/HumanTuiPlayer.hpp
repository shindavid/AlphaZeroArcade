#pragma once

#include "core/BasicTypes.hpp"
#include "games/chess/Game.hpp"
#include "generic_players/HumanTuiPlayer.hpp"

namespace chess {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 private:
  using State = Game::State;
  ActionResponse prompt_for_action(const State&, const ActionMask&, bool) override;
};

}  // namespace chess

#include "inline/games/chess/players/HumanTuiPlayer.inl"
