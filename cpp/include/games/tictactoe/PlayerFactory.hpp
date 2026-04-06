#pragma once

#include "core/PlayerFactory.hpp"
#include "core/players/RemotePlayerProxyGenerator.hpp"
#include "games/tictactoe/Bindings.hpp"  // IWYU pragma: keep
#include "games/tictactoe/Game.hpp"
#include "games/tictactoe/players/HumanTuiPlayerGenerator.hpp"
#include "games/tictactoe/players/PerfectPlayerGenerator.hpp"
#include "generic_players/PlayerGenerator.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"
#include "generic_players/WebPlayer.hpp"
#include "generic_players/WebPlayerGenerator.hpp"
#include "util/MetaProgramming.hpp"

namespace tictactoe {

class PlayerFactory : public core::PlayerFactory<Game> {
 public:
  using base_t = core::PlayerFactory<Game>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    player_subfactory_vec_t result = {
      new core::PlayerSubfactory<tictactoe::HumanTuiPlayerGenerator>(),
      new core::PlayerSubfactory<tictactoe::PerfectPlayerGenerator>(),
      new core::PlayerSubfactory<generic::WebPlayerGenerator<generic::WebPlayer<Game>>>()};
    mp::for_each<typename Bindings::SupportedTraits>([&result]<typename T>() {
      result.push_back(new core::PlayerSubfactory<generic::CompetitionPlayerGenerator<T>>());
      result.push_back(new core::PlayerSubfactory<generic::TrainingPlayerGenerator<T>>());
    });
    result.push_back(new core::PlayerSubfactory<generic::RandomPlayerGenerator<Game>>());
    result.push_back(new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<Game>>());
    return result;
  }
};

}  // namespace tictactoe
