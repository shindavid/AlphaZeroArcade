#pragma once

#include "core/PlayerFactory.hpp"
#include "core/players/RemotePlayerProxyGenerator.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/players/HumanTuiPlayerGenerator.hpp"
#include "games/stochastic_nim/players/PerfectPlayerGenerator.hpp"
#include "generic_players/MctsPlayerGenerator.hpp"
#include "generic_players/RandomPlayerGenerator.hpp"

namespace stochastic_nim {

class PlayerFactory : public core::PlayerFactory<Game> {
 public:
  using base_t = core::PlayerFactory<Game>;
  using player_subfactory_vec_t = base_t::player_subfactory_vec_t;
  using AlphaZeroEvalSpec = core::EvalSpec<Game, core::kParadigmAlphaZero>;

  PlayerFactory() : base_t(make_subfactories()) {}

 private:
  static player_subfactory_vec_t make_subfactories() {
    return {
      new core::PlayerSubfactory<stochastic_nim::HumanTuiPlayerGenerator>(),
      new core::PlayerSubfactory<stochastic_nim::PerfectPlayerGenerator>(),
      new core::PlayerSubfactory<generic::CompetitiveMctsPlayerGenerator<AlphaZeroEvalSpec>>(),
      new core::PlayerSubfactory<generic::TrainingMctsPlayerGenerator<AlphaZeroEvalSpec>>(),
      new core::PlayerSubfactory<generic::RandomPlayerGenerator<Game>>(),
      new core::PlayerSubfactory<core::RemotePlayerProxyGenerator<Game>>()};
  }
};

}  // namespace stochastic_nim
