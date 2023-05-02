#pragma once

#include <common/PlayerFactory.hpp>
#include <common/players/MctsPlayerGenerator.hpp>
#include <common/players/RandomPlayerGenerator.hpp>
#include <common/players/RemotePlayerProxyGenerator.hpp>
#include <othello/GameState.hpp>
#include <othello/Tensorizor.hpp>
#include <othello/players/HumanTuiPlayerGenerator.hpp>

namespace othello {

class PlayerFactory : public common::PlayerFactory<GameState> {
public:
  using base_t = common::PlayerFactory<GameState>;
  using player_generator_creator_vec_t = base_t::player_generator_creator_vec_t;

  PlayerFactory() : base_t(make_generators()) {}

private:
  static player_generator_creator_vec_t make_generators() {
    return {
      new common::PlayerGeneratorCreator<othello::HumanTuiPlayerGenerator>(),
      new common::PlayerGeneratorCreator<common::CompetitiveMctsPlayerGenerator<GameState, Tensorizor>>(),
      new common::PlayerGeneratorCreator<common::TrainingMctsPlayerGenerator<GameState, Tensorizor>>(),
      new common::PlayerGeneratorCreator<common::RandomPlayerGenerator<GameState>>(),
      new common::PlayerGeneratorCreator<common::RemotePlayerProxyGenerator<GameState>>()
    };
  }
};

}  // namespace othello
