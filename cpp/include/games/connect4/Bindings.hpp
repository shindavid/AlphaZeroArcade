#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
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

  using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
};

}  // namespace c4::alpha0

namespace core {

template <>
struct InputTensorizor<c4::Game> : public c4::InputTensorizor {
  using Keys = core::DefaultKeys<c4::Game>;
};

template <>
struct EvalSpec<c4::Game, core::kParadigmAlphaZero> {
  using Game = c4::Game;
  using TrainingTargets = c4::alpha0::TrainingTargets;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<c4::Game, core::kParadigmBetaZero> {
  using Game = c4::Game;
  using TrainingTargets = c4::alpha0::TrainingTargets;
};

}  // namespace core
