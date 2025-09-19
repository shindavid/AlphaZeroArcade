#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/TrainingTargets.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/InputTensorizor.hpp"
#include "util/MetaProgramming.hpp"

namespace c4::alpha0 {

struct TrainingTargets {
  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;
  using OppPolicyTarget = core::OppPolicyTarget<Game>;

  using PrimaryList = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget>;
  using AuxList = mp::TypeList<OppPolicyTarget>;
};

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 10.583;  // likely too big, just keeping previous value
};

}  // namespace c4::alpha0

namespace c4::beta0 {

struct TrainingTargets {
  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;
  using ValueUncertaintyTarget = core::ValueUncertaintyTarget<Game>;
  using ActionValueUncertaintyTarget = core::ActionValueUncertaintyTarget<Game>;
  using OppPolicyTarget = core::OppPolicyTarget<Game>;

  using PrimaryList = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget,
                                   ValueUncertaintyTarget, ActionValueUncertaintyTarget>;
  using AuxList = mp::TypeList<OppPolicyTarget>;
};

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
  using MctsConfiguration = c4::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<c4::Game, core::kParadigmBetaZero> {
  using Game = c4::Game;
  static constexpr SearchParadigm kParadigm = core::kParadigmBetaZero;

  using TrainingTargets = c4::beta0::TrainingTargets;
  using MctsConfiguration = c4::beta0::MctsConfiguration;
};

}  // namespace core
