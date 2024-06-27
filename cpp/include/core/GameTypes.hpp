#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/GameConstants.hpp>
#include <util/EigenUtil.hpp>

#include <array>
#include <bitset>
#include <string>

namespace core {

template<concepts::GameConstants GameConstants, typename BaseState>
struct GameTypes {
  using ActionMask = std::bitset<GameConstants::kNumActions>;
  using player_name_array_t = std::array<std::string, GameConstants::kNumPlayers>;

  using PolicyShape = Eigen::Sizes<GameConstants::kNumActions>;
  using PolicyTensor = eigen_util::FTensor<PolicyShape>;
  using ValueArray = eigen_util::FArray<GameConstants::kNumPlayers>;
  using ActionOutcome = core::ActionOutcome<ValueArray>;

  /*
   * Return type for an MCTS search.
   *
   * This is declared here so that we can properly declare the function signature for the
   * Game::IO::print_mcts_results() function for each specific Game.
   */
  struct SearchResults {
    ActionMask valid_actions;
    PolicyTensor counts;
    PolicyTensor policy_target;
    PolicyTensor policy_prior;
    ValueArray win_rates;
    ValueArray value_prior;
    bool provably_lost = false;
  };

  /*
   * A (symmetry-adjusted) view of a specific position in a game log.
   *
   * This is declared here so that we can properly declare the function signature for the
   * training-target classes for each specific Game.
   */
  struct GameLogView {
    const BaseState* cur_pos;
    const BaseState* final_pos;
    const ValueArray* outcome;
    const PolicyTensor* policy;
    const PolicyTensor* next_policy;
  };
};

}  // namespace core
