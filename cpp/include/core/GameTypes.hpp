#pragma once

#include "core/concepts/GameConstantsConcept.hpp"
#include "core/concepts/PlayerResultConcept.hpp"
#include "util/CompactBitSet.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"
#include "util/Gaussian1D.hpp"

#include <Eigen/Core>

#include <array>
#include <string>

namespace core {

template <concepts::GameConstants GameConstants, typename Move_, typename MoveList_,
          typename State_, concepts::PlayerResult PlayerResult_,
          group::concepts::FiniteGroup SymmetryGroup>
struct GameTypes {
  using Move = Move_;
  using MoveSet = MoveList_;
  using State = State_;
  static constexpr int kNumMoves = GameConstants::kNumMoves;
  static constexpr int kMaxBranchingFactor = GameConstants::kMaxBranchingFactor;
  static constexpr int kNumPlayers = GameConstants::kNumPlayers;

  using PlayerResult = PlayerResult_;
  using GameOutcome = std::array<PlayerResult, kNumPlayers>;

  using player_name_array_t = std::array<std::string, kNumPlayers>;
  using player_bitset_t = util::CompactBitSet<kNumPlayers>;

  using WinShareShape = Eigen::Sizes<kNumPlayers>;
  using WinShareTensor = eigen_util::FTensor<WinShareShape>;
  using ActionValueShape = Eigen::Sizes<kNumMoves, kNumPlayers>;
  using ActionValueTensor = eigen_util::FTensor<ActionValueShape>;

  using LogitValueArray = std::array<util::Gaussian1D, kNumPlayers>;
  using ValueArray = eigen_util::FArray<kNumPlayers>;
  using SymmetryMask = util::CompactBitSet<SymmetryGroup::kOrder>;
  using LocalPolicyArray = eigen_util::DArray<kMaxBranchingFactor>;
  using LocalActionValueArray =
    Eigen::Array<float, Eigen::Dynamic, kNumPlayers, Eigen::RowMajor, kMaxBranchingFactor>;
};

}  // namespace core
