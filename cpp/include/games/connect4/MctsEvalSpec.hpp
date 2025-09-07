#pragma once

#include "core/MctsEvalSpec.hpp"
#include "core/TrainingTargets.hpp"
#include "games/connect4/Game.hpp"
#include "util/MetaProgramming.hpp"

#include <Eigen/Core>

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
