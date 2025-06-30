#pragma once

#include <core/BasicTypes.hpp>
#include <games/stochastic_nim/Game.hpp>
#include <generic_players/HumanTuiPlayer.hpp>

namespace stochastic_nim {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 public:
  using base_t = generic::HumanTuiPlayer<Game>;
  using State = Game::State;

 private:
  core::action_t prompt_for_action(const State&, const ActionMask&) override;
  void print_state(const State&, bool terminal) override;
};

}  // namespace stochastic_nim

#include <inline/games/stochastic_nim/players/HumanTuiPlayer.inl>
