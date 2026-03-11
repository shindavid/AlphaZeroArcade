#pragma once

#include "core/ActionResponse.hpp"
#include "games/chess/Game.hpp"
#include "generic_players/HumanTuiPlayer.hpp"

namespace a0achess {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 private:
  using State = Game::State;
  core::ActionResponse prompt_for_action(const ActionRequest&) override;
};

}  // namespace a0achess

#include "inline/games/chess/players/HumanTuiPlayer.inl"
