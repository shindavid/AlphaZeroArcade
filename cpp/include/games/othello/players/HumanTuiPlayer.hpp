#pragma once

#include "core/BasicTypes.hpp"
#include "games/othello/Game.hpp"
#include "generic_players/HumanTuiPlayer.hpp"

namespace othello {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 private:
  using State = Game::State;
  ActionResponse prompt_for_action(const ActionRequest&) override;
};

}  // namespace othello

#include "inline/games/othello/players/HumanTuiPlayer.inl"
