#pragma once

#include "core/PlayerFactory.hpp"
#include "core/players/RemotePlayerProxyGenerator.hpp"
#include "games/connect4/Bindings.hpp"  // IWYU pragma: keep
#include "games/connect4/Game.hpp"
#include "games/connect4/players/HumanTuiPlayerGenerator.hpp"
#include "games/connect4/players/PerfectPlayerGenerator.hpp"
#include "generic_players/PlayerGenerator.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "generic_players/WebPlayer.hpp"
#include "generic_players/WebPlayerGenerator.hpp"
#include "util/MetaProgramming.hpp"

namespace c4 {

class PlayerFactory : public core::PlayerFactory<Game> {
 public:
  using base_t = core::PlayerFactory<Game>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    player_subfactory_vec_t result = {new core::PlayerSubfactory<c4::HumanTuiPlayerGenerator>()};
    mp::for_each<typename Bindings::SupportedTraits>([&result]<typename T>() {
      result.push_back(new core::PlayerSubfactory<generic::CompetitionPlayerGenerator<T>>());
      result.push_back(new core::PlayerSubfactory<generic::TrainingPlayerGenerator<T>>());
    });
    result.push_back(new core::PlayerSubfactory<c4::PerfectPlayerGenerator>());
    result.push_back(
      new core::PlayerSubfactory<generic::WebPlayerGenerator<generic::WebPlayer<Game>>>());
    result.push_back(new core::PlayerSubfactory<generic::RandomPlayerGenerator<Game>>());
    result.push_back(new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<Game>>());
    return result;
  }
};

}  // namespace c4
