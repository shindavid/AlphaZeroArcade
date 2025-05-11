#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <generic_players/HumanTuiPlayerGenerator.hpp>
#include <games/othello/Game.hpp>
#include <games/othello/players/HumanTuiPlayer.hpp>

namespace othello {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<othello::Game> {
 public:
  using base_t = generic::HumanTuiPlayerGenerator<othello::Game>;
  using base_t::base_t;

  core::AbstractPlayer<othello::Game>* generate(core::game_slot_index_t) override {
    return new HumanTuiPlayer();
  }
};

}  // namespace othello
