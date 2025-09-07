#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/TrainingTargets.hpp"
#include "games/hex/Game.hpp"
#include "games/hex/InputTensorizor.hpp"
#include "util/MetaProgramming.hpp"

namespace hex::alpha0 {

struct TrainingTargets {
    using BoardShape = Eigen::Sizes<Constants::kBoardDim, Constants::kBoardDim>;

    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using ActionValueTarget = core::ActionValueTarget<Game>;
    using OppPolicyTarget = core::OppPolicyTarget<Game>;

    using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
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
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<hex::Game, core::kParadigmBetaZero> {
  using Game = hex::Game;
  using TrainingTargets = hex::alpha0::TrainingTargets;
};

}  // namespace core
