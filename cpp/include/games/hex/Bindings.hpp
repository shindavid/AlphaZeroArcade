#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TrainingTargets.hpp"
#include "games/hex/Game.hpp"
#include "games/hex/InputTensorizor.hpp"

namespace hex::alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<Game>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<Game>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 8;
};

}  // namespace hex::alpha0

namespace core {

template <>
struct InputTensorizor<hex::Game> : public hex::InputTensorizor {
  using Keys = core::DefaultKeys<hex::Game>;
};

template <>
struct EvalSpec<hex::Game, core::kParadigmAlphaZero> {
  using Game = hex::Game;
  using TrainingTargets = hex::alpha0::TrainingTargets;
  using NetworkHeads = hex::alpha0::NetworkHeads;
  using MctsConfiguration = hex::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<hex::Game, core::kParadigmBetaZero> {
  using Game = hex::Game;
  using TrainingTargets = hex::alpha0::TrainingTargets;
  using NetworkHeads = hex::alpha0::NetworkHeads;
  using MctsConfiguration = hex::alpha0::MctsConfiguration;
};

}  // namespace core
