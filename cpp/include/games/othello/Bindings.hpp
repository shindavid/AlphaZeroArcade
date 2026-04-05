#pragma once

#include "alpha0/Traits.hpp"
#include "beta0/Traits.hpp"
#include "core/DefaultTransposer.hpp"
#include "core/EvalSpec.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TensorEncodings.hpp"
#include "core/TrainingTargets.hpp"
#include "core/WinLossDrawEncoding.hpp"
#include "games/othello/Game.hpp"
#include "games/othello/InputEncoder.hpp"
#include "games/othello/InputFrame.hpp"
#include "games/othello/PolicyEncoding.hpp"
#include "games/othello/Symmetries.hpp"
#include "util/MetaProgramming.hpp"

namespace othello {

using GameResultEncoding = core::WinLossDrawEncoding<Game>;
using TensorEncodings =
  core::TensorEncodings<Game, InputEncoder, PolicyEncoding, GameResultEncoding>;

}  // namespace othello

namespace othello::alpha0 {

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

}  // namespace othello::alpha0

namespace othello::beta0 {

struct TrainingTargets {
  using List = mp::Concat_t<core::beta0::StandardTrainingTargetsList<TensorEncodings>,
                            othello::alpha0::TrainingTargets::AuxList>;
};

using NetworkHeads = core::beta0::StandardNetworkHeads<TensorEncodings>;
using MctsConfiguration = alpha0::MctsConfiguration;

}  // namespace othello::beta0

namespace core {

template <>
struct EvalSpec<othello::Game, core::kParadigmAlphaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = othello::Game;
  using InputFrame = othello::InputFrame;
  using Symmetries = othello::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using TensorEncodings = othello::TensorEncodings;
  using TrainingTargets = othello::alpha0::TrainingTargets;
  using NetworkHeads = othello::alpha0::NetworkHeads;
  using MctsConfiguration = othello::alpha0::MctsConfiguration;
};

template <>
struct EvalSpec<othello::Game, core::kParadigmBetaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmBetaZero;
  using Game = othello::Game;
  using InputFrame = othello::InputFrame;
  using Symmetries = othello::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using TensorEncodings = othello::TensorEncodings;
  using TrainingTargets = othello::beta0::TrainingTargets;
  using NetworkHeads = othello::beta0::NetworkHeads;
  using MctsConfiguration = othello::beta0::MctsConfiguration;
};

}  // namespace core

namespace othello {

struct Bindings {
  using SupportedTraits = mp::TypeList<::alpha0::Traits<Game>, ::beta0::Traits<Game>>;
};

}  // namespace othello

#include "inline/games/othello/Bindings.inl"
