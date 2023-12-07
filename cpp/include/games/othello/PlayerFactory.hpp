#pragma once

#include <core/PlayerFactory.hpp>
#include <core/players/RemotePlayerProxyGenerator.hpp>
#include <games/generic/players/MctsPlayerGenerator.hpp>
#include <games/generic/players/RandomPlayerGenerator.hpp>
#include <games/othello/GameState.hpp>
#include <games/othello/Tensorizor.hpp>
#include <games/othello/players/EdaxPlayerGenerator.hpp>
#include <games/othello/players/HumanTuiPlayerGenerator.hpp>

namespace othello {

class PlayerFactory : public core::PlayerFactory<GameState> {
 public:
  using base_t = core::PlayerFactory<GameState>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    return {new core::PlayerSubfactory<othello::HumanTuiPlayerGenerator>(),
            new core::PlayerSubfactory<othello::EdaxPlayerGenerator>(),
            new core::PlayerSubfactory<
                generic::CompetitiveMctsPlayerGenerator<GameState, Tensorizor>>(),
            new core::PlayerSubfactory<
                generic::TrainingMctsPlayerGenerator<GameState, Tensorizor>>(),
            new core::PlayerSubfactory<generic::RandomPlayerGenerator<GameState>>(),
            new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<GameState>>()};
  }
};

}  // namespace othello
