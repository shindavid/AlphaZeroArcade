#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <games/stochastic_nim/Game.hpp>
#include <games/stochastic_nim/players/HumanTuiPlayer.hpp>
#include <generic_players/HumanTuiPlayerGenerator.hpp>

namespace stochastic_nim {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<stochastic_nim::Game> {
 public:
  core::AbstractPlayer<stochastic_nim::Game>* generate(core::game_thread_id_t) override {
    return new stochastic_nim::HumanTuiPlayer();
  }
};

}  // namespace stochastic_nim
