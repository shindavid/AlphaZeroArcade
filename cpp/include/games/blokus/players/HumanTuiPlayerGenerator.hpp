#pragma once

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <generic_players/HumanTuiPlayerGenerator.hpp>
#include <games/blokus/Game.hpp>
#include <games/blokus/players/HumanTuiPlayer.hpp>
#include <util/BoostUtil.hpp>

namespace blokus {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<blokus::Game> {
 public:
  core::AbstractPlayer<blokus::Game>* generate(core::game_thread_id_t) override {
    return new HumanTuiPlayer();
  }
};

}  // namespace blokus
