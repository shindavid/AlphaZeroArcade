#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/TrainingTargets.hpp"
#include "games/nim/Game.hpp"
#include "games/nim/InputTensorizor.hpp"
#include "util/MetaProgramming.hpp"

namespace nim::alpha0 {

struct TrainingTargets {
  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;
  using OppPolicyTarget = core::OppPolicyTarget<Game>;

  using PrimaryList = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget>;
  using AuxList = mp::TypeList<OppPolicyTarget>;
};

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
  using MctsConfiguration = nim::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<nim::Game, core::kParadigmBetaZero> {
  using Game = nim::Game;
  using TrainingTargets = nim::alpha0::TrainingTargets;
  using MctsConfiguration = nim::alpha0::MctsConfiguration;
};

}  // namespace core
