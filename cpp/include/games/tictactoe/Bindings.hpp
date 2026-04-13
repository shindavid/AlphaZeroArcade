#pragma once

#include "core/SearchParadigm.hpp"
#include "core/DefaultTransposer.hpp"
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

namespace tictactoe {

namespace alpha0 {

using GameResultEncoding = core::WinLossDrawEncoding<Game>;
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, GameResultEncoding>;

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

struct Spec {
  static constexpr core::SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = tictactoe::Game;
  using InputFrame = tictactoe::InputFrame;
  using Symmetries = tictactoe::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using TensorEncodings = tictactoe::alpha0::TensorEncodings;
  using TrainingTargets = tictactoe::alpha0::TrainingTargets;
  using NetworkHeads = tictactoe::alpha0::NetworkHeads;
  using MctsConfiguration = tictactoe::alpha0::MctsConfiguration;
};

}  // namespace alpha0

struct Bindings {
  using SupportedSpecs = mp::TypeList<alpha0::Spec>;
};

}  // namespace tictactoe

#include "inline/games/tictactoe/Bindings.inl"
