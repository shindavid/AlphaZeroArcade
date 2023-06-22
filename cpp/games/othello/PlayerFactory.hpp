#pragma once

#include <core/PlayerFactory.hpp>
#include <core/players/RemotePlayerProxyGenerator.hpp>
#include <common/players/MctsPlayerGenerator.hpp>
#include <common/players/RandomPlayerGenerator.hpp>
#include <games/othello/GameState.hpp>
#include <games/othello/Tensorizor.hpp>
#include <games/othello/players/EdaxPlayerGenerator.hpp>
#include <games/othello/players/HumanTuiPlayerGenerator.hpp>

namespace othello {

class PlayerFactory : public core::PlayerFactory<GameState> {
public:
  using base_t = core::PlayerFactory<GameState>;
  using player_generator_creator_vec_t = base_t::player_generator_creator_vec_t;

  PlayerFactory() : base_t(make_generators()) {}

private:
  static player_generator_creator_vec_t make_generators() {
    return {
        new core::PlayerGeneratorCreator<othello::HumanTuiPlayerGenerator>(),
        new core::PlayerGeneratorCreator<othello::EdaxPlayerGenerator>(),
      new core::PlayerGeneratorCreator<common::CompetitiveMctsPlayerGenerator<GameState, Tensorizor>>(),
      new core::PlayerGeneratorCreator<common::TrainingMctsPlayerGenerator<GameState, Tensorizor>>(),
      new core::PlayerGeneratorCreator<common::RandomPlayerGenerator<GameState>>(),
      new core::PlayerGeneratorCreator<core::RemotePlayerProxyGenerator<GameState>>()
    };
  }
};

}  // namespace othello
