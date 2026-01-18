#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TrainingTargets.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/InputTensorizor.hpp"

namespace c4::alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<Game>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<Game>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 10.583;  // likely too big, just keeping previous value
};

}  // namespace c4::alpha0

namespace c4::beta0 {

using TrainingTargets = core::beta0::StandardTrainingTargets<Game>;
using NetworkHeads = core::beta0::StandardNetworkHeads<Game>;
using MctsConfiguration = alpha0::MctsConfiguration;

}  // namespace c4::beta0

namespace core {

template <>
struct InputTensorizor<c4::Game> : public c4::InputTensorizor {
  using Keys = core::DefaultKeys<c4::Game>;
};

template <>
struct EvalSpec<c4::Game, core::kParadigmAlphaZero> {
  using Game = c4::Game;
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;

  using TrainingTargets = c4::alpha0::TrainingTargets;
  using NetworkHeads = c4::alpha0::NetworkHeads;
  using MctsConfiguration = c4::alpha0::MctsConfiguration;
};

template <>
struct EvalSpec<c4::Game, core::kParadigmBetaZero> {
  using Game = c4::Game;
  static constexpr SearchParadigm kParadigm = core::kParadigmBetaZero;

  using TrainingTargets = c4::beta0::TrainingTargets;
  using NetworkHeads = c4::beta0::NetworkHeads;
  using MctsConfiguration = c4::beta0::MctsConfiguration;
};

}  // namespace core
