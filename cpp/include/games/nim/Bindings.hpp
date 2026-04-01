#pragma once

#include "core/DefaultTransposer.hpp"
#include "core/EvalSpec.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TrainingTargets.hpp"
#include "games/nim/Game.hpp"
#include "games/nim/InputFrame.hpp"
#include "games/nim/InputTensorizor.hpp"
#include "games/nim/PolicyEncoding.hpp"
#include "games/nim/Symmetries.hpp"

namespace nim::alpha0 {

using TrainingTargets = core::alpha0::StandardTrainingTargets<Game>;
using NetworkHeads = core::alpha0::StandardNetworkHeads<Game>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 3;
};

}  // namespace nim::alpha0

namespace core {

template <>
struct EvalSpec<nim::Game, core::kParadigmAlphaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = nim::Game;
  using InputFrame = nim::InputFrame;
  using Symmetries = nim::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using InputTensorizor = nim::InputTensorizor;
  using PolicyEncoding = nim::PolicyEncoding;
  using TrainingTargets = nim::alpha0::TrainingTargets;
  using NetworkHeads = nim::alpha0::NetworkHeads;
  using MctsConfiguration = nim::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<nim::Game, core::kParadigmBetaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmBetaZero;
  using Game = nim::Game;
  using InputFrame = nim::InputFrame;
  using Symmetries = nim::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using InputTensorizor = nim::InputTensorizor;
  using PolicyEncoding = nim::PolicyEncoding;
  using TrainingTargets = nim::alpha0::TrainingTargets;
  using NetworkHeads = nim::alpha0::NetworkHeads;
  using MctsConfiguration = nim::alpha0::MctsConfiguration;
};

}  // namespace core
