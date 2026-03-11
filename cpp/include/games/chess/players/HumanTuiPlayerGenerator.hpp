#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/players/HumanTuiPlayer.hpp"
#include "generic_players/HumanTuiPlayerGenerator.hpp"
#include "util/BoostUtil.hpp"

namespace a0achess {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<a0achess::Game> {
 public:
  using base_t = generic::HumanTuiPlayerGenerator<a0achess::Game>;
  using base_t::base_t;

  core::AbstractPlayer<a0achess::Game>* generate(core::game_slot_index_t) override {
    return new HumanTuiPlayer();
  }
};

}  // namespace a0achess
