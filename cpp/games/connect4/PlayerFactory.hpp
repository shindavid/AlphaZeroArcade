#pragma once

#include <core/PlayerFactory.hpp>
#include <core/players/RemotePlayerProxyGenerator.hpp>
#include <common/players/MctsPlayerGenerator.hpp>
#include <common/players/RandomPlayerGenerator.hpp>
#include <games/connect4/GameState.hpp>
#include <games/connect4/Tensorizor.hpp>
#include <games/connect4/players/HumanTuiPlayerGenerator.hpp>
#include <games/connect4/players/MctsPlayerGenerator.hpp>
#include <games/connect4/players/PerfectPlayerGenerator.hpp>

namespace c4 {

class PlayerFactory : public core::PlayerFactory<GameState> {
public:
  using base_t = core::PlayerFactory<GameState>;
  using player_generator_creator_vec_t = base_t::player_generator_creator_vec_t;

  PlayerFactory() : base_t(make_generators()) {}

private:
  static player_generator_creator_vec_t make_generators() {
    return {
      new core::PlayerGeneratorCreator<c4::HumanTuiPlayerGenerator>(),
      new core::PlayerGeneratorCreator<c4::CompetitiveMctsPlayerGenerator>(),
      new core::PlayerGeneratorCreator<c4::TrainingMctsPlayerGenerator>(),
      new core::PlayerGeneratorCreator<c4::PerfectPlayerGenerator>(),
      new core::PlayerGeneratorCreator<common::RandomPlayerGenerator<GameState>>(),
      new core::PlayerGeneratorCreator<core::RemotePlayerProxyGenerator<GameState>>()
    };
  }
};

}  // namespace c4
