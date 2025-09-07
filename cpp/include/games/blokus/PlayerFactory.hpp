#pragma once

#include "core/PlayerFactory.hpp"
#include "core/players/RemotePlayerProxyGenerator.hpp"
#include "games/blokus/Game.hpp"
#include "games/blokus/players/HumanTuiPlayerGenerator.hpp"
#include "generic_players/MctsPlayerGenerator.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"

namespace blokus {

class PlayerFactory : public core::PlayerFactory<Game> {
 public:
  using base_t = core::PlayerFactory<Game>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;
  using AlphaZeroTraits = ::alpha0::Traits<Game>;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    return {
      new core::PlayerSubfactory<blokus::HumanTuiPlayerGenerator>(),
      new core::PlayerSubfactory<generic::CompetitiveMctsPlayerGenerator<AlphaZeroTraits>>(),
      new core::PlayerSubfactory<generic::TrainingMctsPlayerGenerator<AlphaZeroTraits>>(),
      new core::PlayerSubfactory<generic::RandomPlayerGenerator<Game>>(),
      new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<Game>>()};
  }
};

}  // namespace blokus
