#pragma once

#include <core/ActionSymmetryTable.hpp>
#include <core/BasicTypes.hpp>
#include <core/concepts/GameConstants.hpp>
#include <core/concepts/GameResults.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>

#include <array>
#include <bitset>
#include <string>

#include <Eigen/Core>

namespace core {

template <concepts::GameConstants GameConstants, typename State, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
struct GameTypes {
  using ActionMask = std::bitset<GameConstants::kNumActions>;
  using player_name_array_t = std::array<std::string, GameConstants::kNumPlayers>;

  using PolicyShape = Eigen::Sizes<GameConstants::kNumActions>;
  using PolicyTensor = eigen_util::FTensor<PolicyShape>;
  using ValueTensor = GameResults::Tensor;
  using ValueShape = ValueTensor::Dimensions;
  using ActionValueShape = Eigen::Sizes<GameConstants::kNumActions>;
  using ActionValueTensor = eigen_util::FTensor<ActionValueShape>;

  using ValueArray = eigen_util::FArray<GameConstants::kNumPlayers>;
  using SymmetryMask = std::bitset<SymmetryGroup::kOrder>;
  using ActionSymmetryTable = core::ActionSymmetryTable<GameConstants, SymmetryGroup>;
  using LocalPolicyArray = eigen_util::DArray<GameConstants::kMaxBranchingFactor>;
  using LocalActionValueArray = eigen_util::DArray<GameConstants::kMaxBranchingFactor>;

  static_assert(std::is_same_v<ValueArray, typename GameResults::ValueArray>);

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
    PolicyTensor Q;
    PolicyTensor Q_sq;
    ActionValueTensor action_values;
    ValueArray win_rates;
    ValueTensor value_prior;
    ActionSymmetryTable action_symmetry_table;
    bool trivial;  // all actions are symmetrically equivalent
    bool provably_lost = false;

    boost::json::object to_json() const;
  };

  /*
   * A (symmetry-adjusted) view of a specific position in a game log.
   *
   * This is declared here so that we can properly declare the function signature for the
   * training-target classes for each specific Game.
   */
  struct GameLogView {
    const State* cur_pos;
    const State* final_pos;
    const ValueTensor* game_result;
    const PolicyTensor* policy;
    const PolicyTensor* next_policy;
    const ActionValueTensor* action_values;
  };
};

}  // namespace core

#include <inline/core/GameTypes.inl>