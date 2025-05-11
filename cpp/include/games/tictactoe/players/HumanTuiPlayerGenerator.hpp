#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <games/tictactoe/Game.hpp>
#include <games/tictactoe/players/HumanTuiPlayer.hpp>
#include <generic_players/HumanTuiPlayerGenerator.hpp>

namespace tictactoe {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<tictactoe::Game> {
 public:
  using base_t = generic::HumanTuiPlayerGenerator<tictactoe::Game>;
  using base_t::base_t;

  core::AbstractPlayer<tictactoe::Game>* generate(core::game_slot_index_t) override {
    return new tictactoe::HumanTuiPlayer();
  }
};

}  // namespace tictactoe
