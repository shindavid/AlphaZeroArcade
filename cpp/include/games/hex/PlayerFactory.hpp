#pragma once

#include "alphazero/Traits.hpp"
#include "core/PlayerFactory.hpp"
#include "core/players/RemotePlayerProxyGenerator.hpp"
#include "games/hex/Game.hpp"
#include "games/hex/players/HumanTuiPlayerGenerator.hpp"
#include "games/hex/players/WebPlayer.hpp"
#include "generic_players/MctsPlayerGenerator.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "generic_players/WebPlayerGenerator.hpp"

namespace hex {

class PlayerFactory : public core::PlayerFactory<Game> {
 public:
  using base_t = core::PlayerFactory<Game>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;
  using AlphaZeroTraits = ::alpha0::Traits<Game>;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    return {new core::PlayerSubfactory<hex::HumanTuiPlayerGenerator>(),
            new core::PlayerSubfactory<generic::CompetitiveMctsPlayerGenerator<AlphaZeroTraits>>(),
            new core::PlayerSubfactory<generic::TrainingMctsPlayerGenerator<AlphaZeroTraits>>(),
            new core::PlayerSubfactory<generic::WebPlayerGenerator<hex::WebPlayer>>(),
            new core::PlayerSubfactory<generic::RandomPlayerGenerator<Game>>(),
            new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<Game>>()};
  }
};

}  // namespace hex
