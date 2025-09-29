#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TrainingTargets.hpp"
#include "games/tictactoe/Game.hpp"
#include "games/tictactoe/InputTensorizor.hpp"
#include "util/MetaProgramming.hpp"

namespace tictactoe::alpha0 {

struct TrainingTargets {
  using OwnershipShape = Eigen::Sizes<3, kBoardDimension, kBoardDimension>;

  struct OwnershipTarget {
    static constexpr const char* kName = "ownership";
    using Tensor = eigen_util::FTensor<OwnershipShape>;

    template <typename GameLogView>
    static bool tensorize(const GameLogView& view, Tensor&);
  };

  using AuxList = mp::TypeList<OwnershipTarget>;
  using List = mp::Concat_t<core::alpha0::StandardTrainingTargetsList<Game>, AuxList>;
};

using NetworkHeads = core::alpha0::StandardNetworkHeads<Game>;

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
  using NetworkHeads = tictactoe::alpha0::NetworkHeads;
  using MctsConfiguration = tictactoe::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<tictactoe::Game, core::kParadigmBetaZero> {
  using Game = tictactoe::Game;
  using TrainingTargets = tictactoe::alpha0::TrainingTargets;
  using NetworkHeads = tictactoe::alpha0::NetworkHeads;
  using MctsConfiguration = tictactoe::alpha0::MctsConfiguration;
};

}  // namespace core

#include "inline/games/tictactoe/Bindings.inl"
