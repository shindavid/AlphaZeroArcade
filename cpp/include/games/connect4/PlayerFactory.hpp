#pragma once

#include "core/PlayerFactory.hpp"
#include "core/players/RemotePlayerProxyGenerator.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/players/HumanTuiPlayerGenerator.hpp"
#include "games/connect4/players/PerfectPlayerGenerator.hpp"
#include "games/connect4/players/WebPlayer.hpp"
#include "generic_players/MctsPlayerGenerator.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "generic_players/WebPlayerGenerator.hpp"

namespace c4 {

class PlayerFactory : public core::PlayerFactory<Game> {
 public:
  using base_t = core::PlayerFactory<Game>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    return {new core::PlayerSubfactory<c4::HumanTuiPlayerGenerator>(),
            new core::PlayerSubfactory<generic::CompetitiveMctsPlayerGenerator<Game>>(),
            new core::PlayerSubfactory<generic::TrainingMctsPlayerGenerator<Game>>(),
            new core::PlayerSubfactory<c4::PerfectPlayerGenerator>(),
            new core::PlayerSubfactory<generic::WebPlayerGenerator<c4::WebPlayer>>(),
            new core::PlayerSubfactory<generic::RandomPlayerGenerator<Game>>(),
            new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<Game>>()};
  }
};

}  // namespace c4
