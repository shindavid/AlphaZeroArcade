#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TrainingTargets.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/InputTensorizor.hpp"

namespace stochastic_nim::alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<Game>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<Game>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 3;
};

}  // namespace stochastic_nim::alpha0

namespace core {

template <>
struct InputTensorizor<stochastic_nim::Game> : public stochastic_nim::InputTensorizor {
  using Keys = core::DefaultKeys<stochastic_nim::Game>;
};

template <>
struct EvalSpec<stochastic_nim::Game, core::kParadigmAlphaZero> {
  using Game = stochastic_nim::Game;
  using TrainingTargets = stochastic_nim::alpha0::TrainingTargets;
  using NetworkHeads = stochastic_nim::alpha0::NetworkHeads;
  using MctsConfiguration = stochastic_nim::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<stochastic_nim::Game, core::kParadigmBetaZero> {
  using Game = stochastic_nim::Game;
  using TrainingTargets = stochastic_nim::alpha0::TrainingTargets;
  using NetworkHeads = stochastic_nim::alpha0::NetworkHeads;
  using MctsConfiguration = stochastic_nim::alpha0::MctsConfiguration;
};

}  // namespace core
