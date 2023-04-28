#pragma once

#include <common/PlayerFactory.hpp>
#include <common/players/MctsPlayerGenerator.hpp>
#include <common/players/RandomPlayerGenerator.hpp>
#include <common/players/RemotePlayerProxyGenerator.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <connect4/players/C4HumanTuiPlayerGenerator.hpp>
#include <connect4/players/C4MctsPlayerGenerator.hpp>
#include <connect4/players/C4PerfectPlayerGenerator.hpp>

namespace c4 {

class PlayerFactory : public common::PlayerFactory<GameState> {
public:
  using base_t = common::PlayerFactory<GameState>;
  using player_generator_creator_vec_t = base_t::player_generator_creator_vec_t;

  PlayerFactory() : base_t(make_generators()) {}

private:
  static player_generator_creator_vec_t make_generators() {
    return {
      new common::PlayerGeneratorCreator<c4::HumanTuiPlayerGenerator>(),
      new common::PlayerGeneratorCreator<c4::CompetitiveMctsPlayerGenerator>(),
      new common::PlayerGeneratorCreator<c4::TrainingMctsPlayerGenerator>(),
      new common::PlayerGeneratorCreator<c4::PerfectPlayerGenerator>(),
      new common::PlayerGeneratorCreator<common::RandomPlayerGenerator<GameState>>(),
      new common::PlayerGeneratorCreator<common::RemotePlayerProxyGenerator<GameState>>()
    };
  }
};

}  // namespace c4
