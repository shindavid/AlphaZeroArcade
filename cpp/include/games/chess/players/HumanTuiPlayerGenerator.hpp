#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/players/HumanTuiPlayer.hpp"
#include "generic_players/HumanTuiPlayerGenerator.hpp"
#include "util/BoostUtil.hpp"

namespace chess {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<chess::Game> {
 public:
  using base_t = generic::HumanTuiPlayerGenerator<chess::Game>;
  using base_t::base_t;

  core::AbstractPlayer<chess::Game>* generate(core::game_slot_index_t) override {
    return new HumanTuiPlayer();
  }
};

}  // namespace chess
