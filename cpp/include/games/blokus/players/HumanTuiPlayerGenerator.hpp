#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "games/blokus/Game.hpp"
#include "games/blokus/players/HumanTuiPlayer.hpp"
#include "generic_players/HumanTuiPlayerGenerator.hpp"

namespace blokus {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<blokus::Game> {
 public:
  using base_t = generic::HumanTuiPlayerGenerator<blokus::Game>;
  using base_t::base_t;

  core::AbstractPlayer<blokus::Game>* generate(core::game_slot_index_t) override {
    return new HumanTuiPlayer();
  }
};

}  // namespace blokus
