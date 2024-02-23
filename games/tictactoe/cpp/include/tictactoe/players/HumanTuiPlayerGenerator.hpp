#pragma once

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <games/generic/players/HumanTuiPlayerGenerator.hpp>
#include <tictactoe/GameState.hpp>
#include <tictactoe/players/HumanTuiPlayer.hpp>
#include <util/BoostUtil.hpp>

namespace tictactoe {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<tictactoe::GameState> {
 public:
  core::AbstractPlayer<tictactoe::GameState>* generate(core::game_thread_id_t) override {
    return new tictactoe::HumanTuiPlayer();
  }
};

}  // namespace tictactoe
