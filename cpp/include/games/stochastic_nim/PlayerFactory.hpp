#pragma once

#include "core/PlayerFactory.hpp"
#include "core/players/RemotePlayerProxyGenerator.hpp"
#include "games/stochastic_nim/Bindings.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/players/HumanTuiPlayerGenerator.hpp"
#include "games/stochastic_nim/players/PerfectPlayerGenerator.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "util/MetaProgramming.hpp"

namespace stochastic_nim {

class PlayerFactory : public core::PlayerFactory<Game> {
 public:
  using base_t = core::PlayerFactory<Game>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    player_subfactory_vec_t result = {
      new core::PlayerSubfactory<stochastic_nim::HumanTuiPlayerGenerator>(),
      new core::PlayerSubfactory<stochastic_nim::PerfectPlayerGenerator>()};
    mp::for_each<typename Bindings::SupportedSpecs>([&result]<typename Spec>() {
      using Bundle = core::PlayerBundle<Spec::kParadigm>;
      using Player = Bundle::template Player<Spec>;
      using CompGen = Bundle::template CompetitionPlayerGenerator<Player>;
      using TrainGen = Bundle::template TrainingPlayerGenerator<Player>;
      result.push_back(new Bundle::template Subfactory<CompGen>());
      result.push_back(new Bundle::template Subfactory<TrainGen>());
    });
    result.push_back(new core::PlayerSubfactory<generic::RandomPlayerGenerator<Game>>());
    result.push_back(new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<Game>>());
    return result;
  }
};

}  // namespace stochastic_nim
