#pragma once

#include "core/BasicTypes.hpp"
#include "games/hex/Game.hpp"
#include "generic_players/HumanTuiPlayer.hpp"

namespace hex {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 private:
  using State = Game::State;
  ActionResponse prompt_for_action(const ActionRequest&) override;
};

}  // namespace hex

#include "inline/games/hex/players/HumanTuiPlayer.inl"
