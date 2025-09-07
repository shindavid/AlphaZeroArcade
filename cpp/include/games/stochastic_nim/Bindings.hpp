#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/TrainingTargets.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/InputTensorizor.hpp"
#include "util/MetaProgramming.hpp"

namespace stochastic_nim::alpha0 {

struct TrainingTargets {
  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;
  using OppPolicyTarget = core::OppPolicyTarget<Game>;

  using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
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
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<stochastic_nim::Game, core::kParadigmBetaZero> {
  using Game = stochastic_nim::Game;
  using TrainingTargets = stochastic_nim::alpha0::TrainingTargets;
};

}  // namespace core
