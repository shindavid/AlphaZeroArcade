#pragma once

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <common/players/HumanTuiPlayerGenerator.hpp>
#include <games/othello/GameState.hpp>
#include <games/othello/players/HumanTuiPlayer.hpp>
#include <util/BoostUtil.hpp>

namespace othello {

class HumanTuiPlayerGenerator : public common::HumanTuiPlayerGenerator<othello::GameState> {
public:
  core::AbstractPlayer<othello::GameState>* generate(core::game_thread_id_t) override {
    return new HumanTuiPlayer();
  }
};

}  // namespace othello
