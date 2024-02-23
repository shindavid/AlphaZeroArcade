#pragma once

#include <core/PlayerFactory.hpp>
#include <core/players/RemotePlayerProxyGenerator.hpp>
#include <games/generic/players/MctsPlayerGenerator.hpp>
#include <games/generic/players/RandomPlayerGenerator.hpp>
#include <games/connect4/GameState.hpp>
#include <games/connect4/Tensorizor.hpp>
#include <games/connect4/players/HumanTuiPlayerGenerator.hpp>
#include <games/connect4/players/PerfectPlayerGenerator.hpp>

namespace c4 {

class PlayerFactory : public core::PlayerFactory<GameState> {
 public:
  using base_t = core::PlayerFactory<GameState>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    return {new core::PlayerSubfactory<c4::HumanTuiPlayerGenerator>(),
            new core::PlayerSubfactory<
                generic::CompetitiveMctsPlayerGenerator<GameState, Tensorizor>>(),
            new core::PlayerSubfactory<
                generic::TrainingMctsPlayerGenerator<GameState, Tensorizor>>(),
            new core::PlayerSubfactory<c4::PerfectPlayerGenerator>(),
            new core::PlayerSubfactory<generic::RandomPlayerGenerator<GameState>>(),
            new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<GameState>>()};
  }
};

}  // namespace c4
