#pragma once

#include "core/BayesianMctsEvalSpec.hpp"
#include "core/DefaultKeys.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsEvalSpec.hpp"
#include "core/TrainingTargets.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/InputTensorizor.hpp"
#include "util/MetaProgramming.hpp"

/*
 * Connect4 InputTensorizor
 */

namespace core {

template <>
struct InputTensorizor<c4::Game> : public c4::InputTensorizor {
  using Keys = core::DefaultKeys<c4::Game>;
};

}  // namespace core

/*
 * Connect4 AlphaZero
 */

namespace c4::alpha0 {

struct TrainingTargets {
  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;
  using OppPolicyTarget = core::OppPolicyTarget<Game>;

  using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
};

}  // namespace c4::alpha0

namespace core::alpha0 {

template <>
struct EvalSpec<c4::Game> {
  using Game = c4::Game;
  using TrainingTargets = c4::alpha0::TrainingTargets;
};

}  // namespace core::alpha0

/*
 * Connect4 BetaZero: for now, identical to Connect4 AlphaZero.
 */

namespace c4::beta0 {

struct TrainingTargets {
  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;
  using OppPolicyTarget = core::OppPolicyTarget<Game>;

  using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
};

}  // namespace c4::beta0

namespace core::beta0 {

template <>
struct EvalSpec<c4::Game> {
  using Game = c4::Game;
  using TrainingTargets = c4::beta0::TrainingTargets;
};

}  // namespace core::beta0
