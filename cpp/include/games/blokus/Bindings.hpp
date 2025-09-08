#pragma once

#include "core/DefaultKeys.hpp"
#include "core/EvalSpec.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/TrainingTargets.hpp"
#include "games/blokus/Game.hpp"
#include "games/blokus/InputTensorizor.hpp"
#include "util/MetaProgramming.hpp"

namespace blokus::alpha0 {

struct TrainingTargets {
  using GameLogView = blokus::Game::Types::GameLogView;
  using BoardShape = Eigen::Sizes<kBoardDimension, kBoardDimension>;
  using OwnershipShape = Eigen::Sizes<kNumPlayers + 1, kBoardDimension, kBoardDimension>;
  using ScoreShape = Eigen::Sizes<2, kVeryBadScore + 1, kNumPlayers>;  // pdf/cdf, score, player
  using UnplayedPiecesShape = Eigen::Sizes<kNumPlayers, kNumPieces>;

  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;

  struct ScoreTarget {
    static constexpr const char* kName = "score";
    using Tensor = eigen_util::FTensor<ScoreShape>;

    static bool tensorize(const GameLogView& view, Tensor&);
  };

  /*
   * Who owns which square at the end of the game.
   */
  struct OwnershipTarget {
    static constexpr const char* kName = "ownership";
    using Tensor = eigen_util::FTensor<OwnershipShape>;

    static bool tensorize(const GameLogView& view, Tensor&);
  };

  /*
   * Which pieces are unplayed at the end of the game.
   */
  struct UnplayedPiecesTarget {
    static constexpr const char* kName = "unplayed_pieces";
    using Tensor = eigen_util::FTensor<UnplayedPiecesShape>;

    static bool tensorize(const GameLogView& view, Tensor&);
  };

  // TODO:
  // - ReachableSquaresTarget: for each square, whether it is reachable by some player if all
  //                           other players are forced to pass all their turns.
  // - OpponentReplySquaresTarget: for each square, whether some opponent plays a piece there
  //                               before the current player's next move.

  using PrimaryList = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget>;
  using AuxList = mp::TypeList<ScoreTarget, OwnershipTarget, UnplayedPiecesTarget>;
};

struct MctsConfiguration : public core::MctsConfigurationBase {
  static constexpr float kOpeningLength = 70.314;  // likely too big, just keeping previous value
};

}  // namespace blokus::alpha0

namespace core {

template <>
struct InputTensorizor<blokus::Game> : public blokus::InputTensorizor {
  using Keys = core::DefaultKeys<blokus::Game>;
};

template <>
struct EvalSpec<blokus::Game, core::kParadigmAlphaZero> {
  using Game = blokus::Game;
  using TrainingTargets = blokus::alpha0::TrainingTargets;
  using MctsConfiguration = blokus::alpha0::MctsConfiguration;
};

// For now, BetaZero EvalSpec is identical to AlphaZero EvalSpec.
template <>
struct EvalSpec<blokus::Game, core::kParadigmBetaZero> {
  using Game = blokus::Game;
  using TrainingTargets = blokus::alpha0::TrainingTargets;
  using MctsConfiguration = blokus::alpha0::MctsConfiguration;
};

}  // namespace core

#include "inline/games/blokus/Bindings.inl"
