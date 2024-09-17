#pragma once

#include <core/concepts/GameConstants.hpp>
#include <util/EigenUtil.hpp>

#include <Eigen/Core>

namespace core {

template <concepts::GameConstants GameConstants>
struct EigenTypes {
  using PolicyShape = Eigen::Sizes<GameConstants::kNumActions>;
  using PolicyTensor = eigen_util::FTensor<PolicyShape>;
  using ActionValueShape = Eigen::Sizes<GameConstants::kNumActions>;
  using ActionValueTensor = eigen_util::FTensor<ActionValueShape>;
  using ValueArray = eigen_util::FArray<GameConstants::kNumPlayers>;
};

}  // namespace core
