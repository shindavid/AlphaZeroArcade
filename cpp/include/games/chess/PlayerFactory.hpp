#pragma once

#include "alpha0/Traits.hpp"
#include "core/PlayerFactory.hpp"
#include "core/players/RemotePlayerProxyGenerator.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/players/HumanTuiPlayerGenerator.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "generic_players/alpha0/PlayerGenerator.hpp"

namespace chess {

class PlayerFactory : public core::PlayerFactory<Game> {
 public:
  using base_t = core::PlayerFactory<Game>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;
  using AlphaZeroTraits = ::alpha0::Traits<Game>;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    return {
      new core::PlayerSubfactory<chess::HumanTuiPlayerGenerator>(),
      new core::PlayerSubfactory<generic::alpha0::CompetitionPlayerGenerator<AlphaZeroTraits>>(),
      new core::PlayerSubfactory<generic::alpha0::TrainingPlayerGenerator<AlphaZeroTraits>>(),
      new core::PlayerSubfactory<generic::RandomPlayerGenerator<Game>>(),
      new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<Game>>()};
  }
};

}  // namespace chess
