#pragma once

#include "core/BasicTypes.hpp"
#include "games/hex/Game.hpp"
#include "generic_players/HumanTuiPlayer.hpp"

namespace hex {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 private:
  using State = Game::State;
  core::action_t prompt_for_action(const State&, const ActionMask&) override;
};

}  // namespace hex

#include "inline/games/hex/players/HumanTuiPlayer.inl"
