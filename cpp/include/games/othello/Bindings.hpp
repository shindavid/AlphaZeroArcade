#pragma once

#include "core/DefaultTransposer.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/SearchParadigm.hpp"
#include "core/TensorEncodings.hpp"
#include "core/TrainingTargets.hpp"
#include "core/WinLossDrawEncoding.hpp"
#include "games/othello/Game.hpp"
#include "games/othello/InputEncoder.hpp"
#include "games/othello/InputFrame.hpp"
#include "games/othello/PolicyEncoding.hpp"
#include "games/othello/Symmetries.hpp"

namespace othello {

namespace alpha0 {

using GameResultEncoding = core::WinLossDrawEncoding<Game>;
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, GameResultEncoding>;

struct TrainingTargets {
  using OwnershipShape = Eigen::Sizes<3, kBoardDimension, kBoardDimension>;
  using ScoreMarginShape = Eigen::Sizes<2, 2 * kNumCells + 1>;  // pdf/cdf, score-margin

  struct ScoreMarginTarget {
    static constexpr char kName[] = "score_margin";
    using Tensor = eigen_util::FTensor<ScoreMarginShape>;

    template <typename GameLogView>
    static bool encode(const GameLogView& view, Tensor&);
  };

  struct OwnershipTarget {
    static constexpr char kName[] = "ownership";
    using Tensor = eigen_util::FTensor<OwnershipShape>;

    template <typename GameLogView>
    static bool encode(const GameLogView& view, Tensor&);
  };

  using AuxList = mp::TypeList<ScoreMarginTarget, OwnershipTarget>;
  using List = mp::Concat_t<core::alpha0::StandardTrainingTargetsList<TensorEncodings>, AuxList>;
};

using NetworkHeads = core::alpha0::StandardNetworkHeads<TensorEncodings>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 25.298;  // likely too big, just keeping previous value
};

struct Spec {
  static constexpr core::SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = othello::Game;
  using InputFrame = othello::InputFrame;
  using Symmetries = othello::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using TensorEncodings = othello::alpha0::TensorEncodings;
  using TrainingTargets = othello::alpha0::TrainingTargets;
  using NetworkHeads = othello::alpha0::NetworkHeads;
  using MctsConfiguration = othello::alpha0::MctsConfiguration;
};

}  // namespace alpha0

struct Bindings {
  using SupportedSpecs = mp::TypeList<alpha0::Spec>;
};

}  // namespace othello

#include "inline/games/othello/Bindings.inl"
