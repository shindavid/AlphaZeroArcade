#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TrainingTargets.hpp"
#include "games/nim/Game.hpp"
#include "games/nim/InputTensorizor.hpp"

namespace nim::alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<Game>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<Game>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 3;
};

}  // namespace nim::alpha0

namespace core {

template <>
struct InputTensorizor<nim::Game> : public nim::InputTensorizor {
  using Keys = core::DefaultKeys<nim::Game>;
};

template <>
struct EvalSpec<nim::Game, core::kParadigmAlphaZero> {
  using Game = nim::Game;
  using TrainingTargets = nim::alpha0::TrainingTargets;
  using NetworkHeads = nim::alpha0::NetworkHeads;
  using MctsConfiguration = nim::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<nim::Game, core::kParadigmBetaZero> {
  using Game = nim::Game;
  using TrainingTargets = nim::alpha0::TrainingTargets;
  using NetworkHeads = nim::alpha0::NetworkHeads;
  using MctsConfiguration = nim::alpha0::MctsConfiguration;
};

}  // namespace core
