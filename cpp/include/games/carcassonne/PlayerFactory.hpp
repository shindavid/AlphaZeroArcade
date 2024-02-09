#pragma once

#include <core/PlayerFactory.hpp>
#include <core/players/RemotePlayerProxyGenerator.hpp>
#include <games/generic/players/MctsPlayerGenerator.hpp>
#include <games/generic/players/RandomPlayerGenerator.hpp>
#include <games/carcassonne/GameState.hpp>
#include <games/carcassonne/Tensorizor.hpp>
#include <games/carcassonne/players/HumanTuiPlayerGenerator.hpp>
#include <games/carcassonne/players/PerfectPlayerGenerator.hpp>

namespace carcassonne {

class PlayerFactory : public core::PlayerFactory<GameState> {
 public:
  using base_t = core::PlayerFactory<GameState>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    return {
        new core::PlayerSubfactory<carcassonne::HumanTuiPlayerGenerator>(),
        new core::PlayerSubfactory<carcassonne::PerfectPlayerGenerator>(),
        new core::PlayerSubfactory<
            generic::CompetitiveMctsPlayerGenerator<GameState, Tensorizor>>(),
        new core::PlayerSubfactory<generic::TrainingMctsPlayerGenerator<GameState, Tensorizor>>(),
        new core::PlayerSubfactory<generic::RandomPlayerGenerator<GameState>>(),
        new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<GameState>>()};
  }
};

}  // namespace carcassonne
