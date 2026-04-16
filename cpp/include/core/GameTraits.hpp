#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConstantsConcept.hpp"
#include "core/concepts/PlayerResultConcept.hpp"
#include "util/CompactBitSet.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"
#include "util/Gaussian1D.hpp"

#include <Eigen/Core>

#include <array>
#include <concepts>
#include <string>

namespace core {

template <concepts::GameConstants GameConstants, typename Move_, typename MoveList_,
          typename State_, typename InfoSet_, concepts::PlayerResult PlayerResult_,
          group::concepts::FiniteGroup SymmetryGroup>
struct GameTraits {
  using Move = Move_;
  using MoveSet = MoveList_;
  using State = State_;
  using InfoSet = InfoSet_;
  static constexpr information_level_t kInformationLevel =
    std::same_as<State, InfoSet> ? kPerfectInfo : kImperfectInfo;
  static constexpr int kNumMoves = GameConstants::kNumMoves;
  static constexpr int kMaxBranchingFactor = GameConstants::kMaxBranchingFactor;
  static constexpr int kNumPlayers = GameConstants::kNumPlayers;

  using PlayerResult = PlayerResult_;
  using GameOutcome = std::array<PlayerResult, kNumPlayers>;

  using player_name_array_t = std::array<std::string, kNumPlayers>;
  using player_bitset_t = util::CompactBitSet<kNumPlayers>;

  using LogitValueArray = std::array<util::Gaussian1D, kNumPlayers>;
  using ValueArray = eigen_util::FArray<kNumPlayers>;
  using SymmetryMask = util::CompactBitSet<SymmetryGroup::kOrder>;
  using LocalPolicyArray = eigen_util::DArray<kMaxBranchingFactor>;
  using LocalActionValueArray =
    Eigen::Array<float, Eigen::Dynamic, kNumPlayers, Eigen::RowMajor, kMaxBranchingFactor>;
};

}  // namespace core
