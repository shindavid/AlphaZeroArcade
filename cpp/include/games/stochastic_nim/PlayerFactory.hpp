#pragma once

#include <core/PlayerFactory.hpp>
#include <core/players/RemotePlayerProxyGenerator.hpp>
#include <generic_players/MctsPlayerGenerator.hpp>
#include <generic_players/RandomPlayerGenerator.hpp>
#include <games/stochastic_nim/Game.hpp>
#include <games/stochastic_nim/PerfectPlayerGenerator.hpp>

namespace stochastic_nim {

class PlayerFactory : public core::PlayerFactory<Game> {
 public:
  using base_t = core::PlayerFactory<Game>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    return {new core::PlayerSubfactory<stochastic_nim::PerfectPlayerGenerator>(),
            new core::PlayerSubfactory<generic::CompetitiveMctsPlayerGenerator<Game>>(),
            new core::PlayerSubfactory<generic::TrainingMctsPlayerGenerator<Game>>(),
            new core::PlayerSubfactory<generic::RandomPlayerGenerator<Game>>(),
            new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<Game>>()};
  }
};

}  // namespace stochastic_nim
