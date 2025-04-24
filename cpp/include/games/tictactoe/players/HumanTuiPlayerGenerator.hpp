#pragma once

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <games/tictactoe/Game.hpp>
#include <games/tictactoe/players/HumanTuiPlayer.hpp>
#include <generic_players/HumanTuiPlayerGenerator.hpp>
#include <util/BoostUtil.hpp>

namespace tictactoe {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<tictactoe::Game> {
 public:
  core::AbstractPlayer<tictactoe::Game>* generate(core::game_slot_index_t) override {
    return new tictactoe::HumanTuiPlayer();
  }
};

}  // namespace tictactoe
