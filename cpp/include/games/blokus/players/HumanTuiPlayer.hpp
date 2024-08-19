#pragma once

#include <core/BasicTypes.hpp>
#include <generic_players/HumanTuiPlayer.hpp>
#include <games/blokus/Game.hpp>

namespace blokus {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 private:
  using FullState = Game::FullState;
  core::action_t prompt_for_action(const FullState&, const ActionMask&) override;
};

}  // namespace blokus

#include <inline/games/blokus/players/HumanTuiPlayer.inl>
