#pragma once

#include "core/PlayerFactory.hpp"
#include "core/players/RemotePlayerProxyGenerator.hpp"
#include "games/othello/Bindings.hpp"
#include "games/othello/Game.hpp"
#include "games/othello/players/EdaxPlayerGenerator.hpp"
#include "games/othello/players/HumanTuiPlayerGenerator.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "generic_players/WebPlayer.hpp"
#include "generic_players/WebPlayerGenerator.hpp"
#include "util/MetaProgramming.hpp"

namespace othello {

class PlayerFactory : public core::PlayerFactory<Game> {
 public:
  using base_t = core::PlayerFactory<Game>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    player_subfactory_vec_t result = {
      new core::PlayerSubfactory<othello::HumanTuiPlayerGenerator>(),
      new core::PlayerSubfactory<othello::EdaxPlayerGenerator>(),
      new core::PlayerSubfactory<generic::WebPlayerGenerator<generic::WebPlayer<Game>>>()};
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

}  // namespace othello
