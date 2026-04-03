#pragma once

#include "alpha0/Traits.hpp"
#include "core/DefaultTransposer.hpp"
#include "core/EvalSpec.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/NetworkHeads.hpp"
#include "core/TrainingTargets.hpp"
#include "games/blokus/Game.hpp"
#include "games/blokus/InputFrame.hpp"
#include "games/blokus/InputTensorizor.hpp"
#include "games/blokus/PolicyEncoding.hpp"
#include "games/blokus/Symmetries.hpp"
#include "util/MetaProgramming.hpp"

namespace blokus::alpha0 {

struct TrainingTargets {
  using BoardShape = Eigen::Sizes<kBoardDimension, kBoardDimension>;
  using OwnershipShape = Eigen::Sizes<kNumPlayers + 1, kBoardDimension, kBoardDimension>;
  using ScoreShape = Eigen::Sizes<2, kVeryBadScore + 1, kNumPlayers>;  // pdf/cdf, score, player
  using UnplayedPiecesShape = Eigen::Sizes<kNumPlayers, kNumPieces>;

  struct ScoreTarget {
    static constexpr char kName[] = "score";
    using Tensor = eigen_util::FTensor<ScoreShape>;

    template <typename GameLogView>
    static bool tensorize(const GameLogView& view, Tensor&);
  };

  /*
   * Who owns which square at the end of the game.
   */
  struct OwnershipTarget {
    static constexpr char kName[] = "ownership";
    using Tensor = eigen_util::FTensor<OwnershipShape>;

    template <typename GameLogView>
    static bool tensorize(const GameLogView& view, Tensor&);
  };

  /*
   * Which pieces are unplayed at the end of the game.
   */
  struct UnplayedPiecesTarget {
    static constexpr char kName[] = "unplayed_pieces";
    using Tensor = eigen_util::FTensor<UnplayedPiecesShape>;

    template <typename GameLogView>
    static bool tensorize(const GameLogView& view, Tensor&);
  };

  // TODO:
  // - ReachableSquaresTarget: for each square, whether it is reachable by some player if all
  //                           other players are forced to pass all their turns.
  // - OpponentReplySquaresTarget: for each square, whether some opponent plays a piece there
  //                               before the current player's next move.

  using AuxList = mp::TypeList<ScoreTarget, OwnershipTarget, UnplayedPiecesTarget>;
  using List = mp::Concat_t<core::alpha0::StandardTrainingTargetsList<Game>, AuxList>;
};

using NetworkHeads = core::alpha0::StandardNetworkHeads<Game>;

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 70.314;  // likely too big, just keeping previous value
};

}  // namespace blokus::alpha0

namespace core {

template <>
struct EvalSpec<blokus::Game, core::kParadigmAlphaZero> {
  static constexpr SearchParadigm kParadigm = core::kParadigmAlphaZero;
  using Game = blokus::Game;
  using InputFrame = blokus::InputFrame;
  using Symmetries = blokus::Symmetries;
  using Transposer = core::DefaultTransposer<Game>;
  using InputTensorizor = blokus::InputTensorizor;
  using PolicyEncoding = blokus::PolicyEncoding;
  using TrainingTargets = blokus::alpha0::TrainingTargets;
  using NetworkHeads = blokus::alpha0::NetworkHeads;
  using MctsConfiguration = blokus::alpha0::MctsConfiguration;
};

}  // namespace core

namespace blokus {

struct Bindings {
  using SupportedTraits = mp::TypeList<::alpha0::Traits<Game>>;
};

}  // namespace blokus

#include "inline/games/blokus/Bindings.inl"
