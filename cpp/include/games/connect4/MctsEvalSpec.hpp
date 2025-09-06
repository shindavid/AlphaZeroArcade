#pragma once

#include "core/DefaultKeys.hpp"
#include "core/MctsEvalSpec.hpp"
#include "core/TrainingTargets.hpp"
#include "games/connect4/Game.hpp"
#include "util/MetaProgramming.hpp"

#include <Eigen/Core>

namespace c4::mcts {

struct TrainingTargets {
  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;
  using OppPolicyTarget = core::OppPolicyTarget<Game>;

  using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
};

}  // namespace c4::mcts

namespace core::mcts {

template <>
struct EvalSpec<c4::Game> {
  using Game = c4::Game;
  using TrainingTargets = c4::mcts::TrainingTargets;
  using Keys = core::DefaultKeys<Game>;
};

}  // namespace core::mcts
