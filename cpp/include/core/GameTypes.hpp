#pragma once

#include "core/GameDerivedConstants.hpp"
#include "core/concepts/GameConstantsConcept.hpp"
#include "core/concepts/GameResultsConcept.hpp"
#include "util/CompactBitSet.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"
#include "util/Gaussian1D.hpp"

#include <Eigen/Core>

#include <array>
#include <string>

namespace core {

template <concepts::GameConstants GameConstants, typename Move_, typename MoveList_,
          typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
struct GameTypes {
  using DerivedConstants = core::DerivedConstants<GameConstants>;

  using Move = Move_;
  using MoveList = MoveList_;
  using State = State_;
  static constexpr int kNumMoves = GameConstants::kNumMoves;
  static constexpr int kMaxBranchingFactor = GameConstants::kMaxBranchingFactor;
  static constexpr int kNumPlayers = GameConstants::kNumPlayers;

  using player_name_array_t = std::array<std::string, kNumPlayers>;
  using player_bitset_t = util::CompactBitSet<kNumPlayers>;

  // TODO: policy encoding should be moved to a separate class.
  using PolicyShape = Eigen::Sizes<kNumMoves>;
  using PolicyTensor = eigen_util::FTensor<PolicyShape>;
  using GameResultTensor = GameResults::Tensor;
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

  static_assert(std::is_same_v<ValueArray, typename GameResults::ValueArray>);
};

}  // namespace core
