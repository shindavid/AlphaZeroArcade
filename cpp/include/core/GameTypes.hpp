#pragma once

#include "core/ActionSymmetryTable.hpp"
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

// TODO: some of the classes whose definitions are inlined here don't need to be. Move them out.
template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
struct GameTypes {
  using DerivedConstants = core::DerivedConstants<GameConstants>;

  using State = State_;
  using kNumActionsPerMode = GameConstants::kNumActionsPerMode;
  static constexpr int kNumActionModes = DerivedConstants::kNumActionModes;
  static constexpr int kMaxNumActions = DerivedConstants::kMaxNumActions;
  static constexpr int kMaxBranchingFactor = GameConstants::kMaxBranchingFactor;
  static constexpr int kNumPlayers = GameConstants::kNumPlayers;

  using ActionMask = util::CompactBitSet<kMaxNumActions>;
  using player_name_array_t = std::array<std::string, kNumPlayers>;
  using player_bitset_t = util::CompactBitSet<kNumPlayers>;

  using PolicyShape = Eigen::Sizes<kMaxNumActions>;
  using PolicyTensor = eigen_util::FTensor<PolicyShape>;
  using GameResultTensor = GameResults::Tensor;
  using WinShareShape = Eigen::Sizes<kNumPlayers>;
  using WinShareTensor = eigen_util::FTensor<WinShareShape>;
  using ActionValueShape = Eigen::Sizes<kMaxNumActions, kNumPlayers>;
  using ActionValueTensor = eigen_util::FTensor<ActionValueShape>;
  using ChanceEventShape = Eigen::Sizes<kMaxNumActions>;
  using ChanceDistribution = eigen_util::FTensor<ChanceEventShape>;

  using LogitValueArray = std::array<util::Gaussian1D, kNumPlayers>;
  using ValueArray = eigen_util::FArray<kNumPlayers>;
  using SymmetryMask = util::CompactBitSet<SymmetryGroup::kOrder>;
  using ActionSymmetryTable = core::ActionSymmetryTable<kMaxNumActions, SymmetryGroup>;
  using LocalPolicyArray = eigen_util::DArray<kMaxBranchingFactor>;
  using LocalActionValueArray =
    Eigen::Array<float, Eigen::Dynamic, kNumPlayers, Eigen::RowMajor, kMaxBranchingFactor>;

  static_assert(std::is_same_v<ValueArray, typename GameResults::ValueArray>);
};

}  // namespace core
