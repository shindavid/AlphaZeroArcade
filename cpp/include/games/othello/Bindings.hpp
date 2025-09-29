#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TrainingTargets.hpp"
#include "games/othello/Game.hpp"
#include "games/othello/InputTensorizor.hpp"
#include "util/MetaProgramming.hpp"

namespace othello::alpha0 {

struct TrainingTargets {
  using OwnershipShape = Eigen::Sizes<3, kBoardDimension, kBoardDimension>;
  using ScoreMarginShape = Eigen::Sizes<2, 2 * kNumCells + 1>;  // pdf/cdf, score-margin

  struct ScoreMarginTarget {
    static constexpr const char* kName = "score_margin";
    using Tensor = eigen_util::FTensor<ScoreMarginShape>;

    template <typename GameLogView>
    static bool tensorize(const GameLogView& view, Tensor&);
  };

  struct OwnershipTarget {
    static constexpr const char* kName = "ownership";
    using Tensor = eigen_util::FTensor<OwnershipShape>;

    template <typename GameLogView>
    static bool tensorize(const GameLogView& view, Tensor&);
  };

  using AuxList = mp::TypeList<ScoreMarginTarget, OwnershipTarget>;
  using List = mp::Concat_t<core::alpha0::StandardTrainingTargetsList<Game>, AuxList>;
};

using NetworkHeads = core::alpha0::StandardNetworkHeads<Game>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 25.298;  // likely too big, just keeping previous value
};

}  // namespace othello::alpha0

namespace core {

template <>
struct InputTensorizor<othello::Game> : public othello::InputTensorizor {
  using Keys = core::DefaultKeys<othello::Game>;
};

template <>
struct EvalSpec<othello::Game, core::kParadigmAlphaZero> {
  using Game = othello::Game;
  using TrainingTargets = othello::alpha0::TrainingTargets;
  using NetworkHeads = othello::alpha0::NetworkHeads;
  using MctsConfiguration = othello::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<othello::Game, core::kParadigmBetaZero> {
  using Game = othello::Game;
  using TrainingTargets = othello::alpha0::TrainingTargets;
  using NetworkHeads = othello::alpha0::NetworkHeads;
  using MctsConfiguration = othello::alpha0::MctsConfiguration;
};

}  // namespace core

#include "inline/games/othello/Bindings.inl"
