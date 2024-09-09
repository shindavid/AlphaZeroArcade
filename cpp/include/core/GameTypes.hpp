#pragma once

#include <core/ActionSymmetryTable.hpp>
#include <core/BasicTypes.hpp>
#include <core/concepts/GameConstants.hpp>
#include <core/EigenTypes.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>

#include <array>
#include <bitset>
#include <string>

#include <Eigen/Core>

namespace core {

template <concepts::GameConstants GameConstants, typename BaseState,
          group::concepts::FiniteGroup Group>
struct GameTypes {
  using ActionMask = std::bitset<GameConstants::kNumActions>;
  using player_name_array_t = std::array<std::string, GameConstants::kNumPlayers>;

  using PolicyShape = EigenTypes<GameConstants>::PolicyShape;
  using PolicyTensor = EigenTypes<GameConstants>::PolicyTensor;
  using ActionValueShape = EigenTypes<GameConstants>::ActionValueShape;
  using ActionValueTensor = EigenTypes<GameConstants>::ActionValueTensor;
  using ValueArray = EigenTypes<GameConstants>::ValueArray;
  using ActionOutcome = core::ActionOutcome<ValueArray>;
  using SymmetryMask = std::bitset<Group::kOrder>;
  using ActionSymmetryTable = core::ActionSymmetryTable<GameConstants, Group>;
  using LocalPolicyArray = eigen_util::DArray<GameConstants::kMaxBranchingFactor>;
  using LocalActionValueArray = eigen_util::DArray<GameConstants::kMaxBranchingFactor>;

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
    ActionValueTensor action_values;
    ValueArray win_rates;
    ValueArray value_prior;
    ActionSymmetryTable action_symmetry_table;
    bool trivial;  // all actions are symmetrically equivalent
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
    const ActionValueTensor* action_values;
  };
};

}  // namespace core
