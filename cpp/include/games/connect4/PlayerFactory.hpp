#pragma once

#include "alpha0/Traits.hpp"
#include "beta0/Traits.hpp"
#include "core/PlayerFactory.hpp"
#include "core/players/RemotePlayerProxyGenerator.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/players/HumanTuiPlayerGenerator.hpp"
#include "games/connect4/players/PerfectPlayerGenerator.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "generic_players/WebPlayer.hpp"
#include "generic_players/WebPlayerGenerator.hpp"
#include "generic_players/alpha0/PlayerGenerator.hpp"
#include "generic_players/beta0/PlayerGenerator.hpp"

namespace c4 {

class PlayerFactory : public core::PlayerFactory<Game> {
 public:
  using base_t = core::PlayerFactory<Game>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;
  using AlphaZeroTraits = ::alpha0::Traits<Game>;
  using BetaZeroTraits = ::beta0::Traits<Game>;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    return {
      new core::PlayerSubfactory<c4::HumanTuiPlayerGenerator>(),
      new core::PlayerSubfactory<generic::alpha0::CompetitionPlayerGenerator<AlphaZeroTraits>>(),
      new core::PlayerSubfactory<generic::alpha0::TrainingPlayerGenerator<AlphaZeroTraits>>(),
      new core::PlayerSubfactory<generic::beta0::CompetitionPlayerGenerator<BetaZeroTraits>>(),
      new core::PlayerSubfactory<generic::beta0::TrainingPlayerGenerator<BetaZeroTraits>>(),
      new core::PlayerSubfactory<c4::PerfectPlayerGenerator>(),
      new core::PlayerSubfactory<generic::WebPlayerGenerator<generic::WebPlayer<Game>>>(),
      new core::PlayerSubfactory<generic::RandomPlayerGenerator<Game>>(),
      new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<Game>>()};
  }
};

}  // namespace c4
