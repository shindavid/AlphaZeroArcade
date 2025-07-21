#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "generic_players/HumanTuiPlayerGenerator.hpp"
#include "games/hex/Game.hpp"
#include "games/hex/players/HumanTuiPlayer.hpp"

namespace hex {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<hex::Game> {
 public:
  using base_t = generic::HumanTuiPlayerGenerator<hex::Game>;
  using base_t::base_t;

  core::AbstractPlayer<hex::Game>* generate(core::game_slot_index_t) override {
    return new HumanTuiPlayer();
  }
};

}  // namespace hex
