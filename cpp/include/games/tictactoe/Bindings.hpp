#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/TrainingTargets.hpp"
#include "games/tictactoe/Game.hpp"
#include "games/tictactoe/InputTensorizor.hpp"
#include "util/MetaProgramming.hpp"

namespace tictactoe::alpha0 {

struct TrainingTargets {
  using BoardShape = Eigen::Sizes<kBoardDimension, kBoardDimension>;
  using OwnershipShape = Eigen::Sizes<3, kBoardDimension, kBoardDimension>;

  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;
  using OppPolicyTarget = core::OppPolicyTarget<Game>;

  struct OwnershipTarget {
    static constexpr const char* kName = "ownership";
    using Tensor = eigen_util::FTensor<OwnershipShape>;

    static bool tensorize(const Game::Types::GameLogView& view, Tensor&);
  };

  using PrimaryList = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget>;
  using AuxList = mp::TypeList<OppPolicyTarget, OwnershipTarget>;
};

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 4;
};

}  // namespace tictactoe::alpha0

namespace core {

template <>
struct InputTensorizor<tictactoe::Game> : public tictactoe::InputTensorizor {
  using Keys = core::DefaultKeys<tictactoe::Game>;
};

template <>
struct EvalSpec<tictactoe::Game, core::kParadigmAlphaZero> {
  using Game = tictactoe::Game;
  using TrainingTargets = tictactoe::alpha0::TrainingTargets;
  using MctsConfiguration = tictactoe::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<tictactoe::Game, core::kParadigmBetaZero> {
  using Game = tictactoe::Game;
  using TrainingTargets = tictactoe::alpha0::TrainingTargets;
  using MctsConfiguration = tictactoe::alpha0::MctsConfiguration;
};

}  // namespace core

#include "inline/games/tictactoe/Bindings.inl"
