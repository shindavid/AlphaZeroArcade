#pragma once

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <generic_players/HumanTuiPlayerGenerator.hpp>
#include <util/BoostUtil.hpp>

#include <games/chess/Game.hpp>
#include <games/chess/players/HumanTuiPlayer.hpp>

namespace chess {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<chess::Game> {
 public:
  core::AbstractPlayer<chess::Game>* generate(core::game_slot_index_t) override {
    return new HumanTuiPlayer();
  }
};

}  // namespace chess

