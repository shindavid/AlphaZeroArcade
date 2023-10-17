#pragma once

#include <core/PlayerFactory.hpp>
#include <core/players/RemotePlayerProxyGenerator.hpp>
#include <common/players/MctsPlayerGenerator.hpp>
#include <common/players/RandomPlayerGenerator.hpp>
#include <games/tictactoe/GameState.hpp>
#include <games/tictactoe/Tensorizor.hpp>
#include <games/tictactoe/players/HumanTuiPlayerGenerator.hpp>
#include <games/tictactoe/players/PerfectPlayerGenerator.hpp>

namespace tictactoe {

class PlayerFactory : public core::PlayerFactory<GameState> {
 public:
  using base_t = core::PlayerFactory<GameState>;
  using player_generator_creator_vec_t = base_t::player_generator_creator_vec_t;

  PlayerFactory() : base_t(make_generators()) {}

 private:
  static player_generator_creator_vec_t make_generators() {
    return {new core::PlayerGeneratorCreator<tictactoe::HumanTuiPlayerGenerator>(),
            new core::PlayerGeneratorCreator<tictactoe::PerfectPlayerGenerator>(),
            new core::PlayerGeneratorCreator<
                common::CompetitiveMctsPlayerGenerator<GameState, Tensorizor>>(),
            new core::PlayerGeneratorCreator<
                common::TrainingMctsPlayerGenerator<GameState, Tensorizor>>(),
            new core::PlayerGeneratorCreator<common::RandomPlayerGenerator<GameState>>(),
            new core::PlayerGeneratorCreator<core::RemotePlayerProxyGenerator<GameState>>()};
  }
};

}  // namespace tictactoe
