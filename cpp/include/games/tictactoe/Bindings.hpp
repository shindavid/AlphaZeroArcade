#pragma once

#include "alpha0/Traits.hpp"
#include "core/DefaultTransposer.hpp"
#include "core/EvalSpec.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TensorEncodings.hpp"
#include "core/TrainingTargets.hpp"
#include "core/WinLossDrawEncoding.hpp"
#include "games/tictactoe/Game.hpp"
#include "games/tictactoe/InputEncoder.hpp"
#include "games/tictactoe/InputFrame.hpp"
#include "games/tictactoe/PolicyEncoding.hpp"
#include "games/tictactoe/Symmetries.hpp"
#include "util/MetaProgramming.hpp"

namespace tictactoe {
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, core::WinLossDrawEncoding>;
}

namespace tictactoe::alpha0 {

struct TrainingTargets {
  using OwnershipShape = Eigen::Sizes<3, kBoardDimension, kBoardDimension>;

  struct OwnershipTarget {
    static constexpr char kName[] = "ownership";
    using Tensor = eigen_util::FTensor<OwnershipShape>;

    template <typename GameLogView>
    static bool encode(const GameLogView& view, Tensor&);
  };

  using AuxList = mp::TypeList<OwnershipTarget>;
  using List = mp::Concat_t<core::alpha0::StandardTrainingTargetsList<TensorEncodings>, AuxList>;
};

using NetworkHeads = core::alpha0::StandardNetworkHeads<TensorEncodings>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 4;
};

}  // namespace tictactoe::alpha0

namespace core {

template <>
struct EvalSpec<tictactoe::Game, core::kParadigmAlphaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = tictactoe::Game;
  using InputFrame = tictactoe::InputFrame;
  using Symmetries = tictactoe::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using TensorEncodings = tictactoe::TensorEncodings;
  using TrainingTargets = tictactoe::alpha0::TrainingTargets;
  using NetworkHeads = tictactoe::alpha0::NetworkHeads;
  using MctsConfiguration = tictactoe::alpha0::MctsConfiguration;
};

}  // namespace core

namespace tictactoe {

struct Bindings {
  using SupportedTraits = mp::TypeList<::alpha0::Traits<Game>>;
};

}  // namespace tictactoe

#include "inline/games/tictactoe/Bindings.inl"
