#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/players/HumanTuiPlayer.hpp"
#include "generic_players/HumanTuiPlayerGenerator.hpp"

namespace stochastic_nim {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<stochastic_nim::Game> {
 public:
  using base_t = generic::HumanTuiPlayerGenerator<stochastic_nim::Game>;
  using base_t::base_t;

  core::AbstractPlayer<stochastic_nim::Game>* generate(core::game_slot_index_t) override {
    return new stochastic_nim::HumanTuiPlayer();
  }
};

}  // namespace stochastic_nim
