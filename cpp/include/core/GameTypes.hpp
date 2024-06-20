#pragma once

#include <core/BasicTypes.hpp>
#include <core/SearchResults.hpp>
#include <util/EigenUtil.hpp>

#include <array>
#include <bitset>
#include <string>

namespace core {

template<typename Game>
struct GameTypes {
  using ActionMask = std::bitset<Game::Constants::kNumActions>;
  using player_name_array_t = std::array<std::string, Game::Constants::kNumPlayers>;

  using PolicyShape = Eigen::Sizes<Game::Constants::kNumActions>;
  using PolicyTensor = eigen_util::FTensor<PolicyShape>;
  using ValueArray = eigen_util::FArray<Game::Constants::kNumPlayers>;
  using ActionOutcome = core::ActionOutcome<ValueArray>;
  using SearchResults = core::SearchResults<GameTypes>;
  using GameLogView = core::GameLogView<Game>;
};

}  // namespace core
