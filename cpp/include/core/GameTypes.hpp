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

template <concepts::GameConstants GameConstants, typename StateType, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
struct GameTypes {
  using State = StateType;
  using kNumActionsPerMode = GameConstants::kNumActionsPerMode;
  static constexpr int kNumActionModes = kNumActionsPerMode::size();
  static constexpr int kMaxNumActions = mp::MaxOf_v<kNumActionsPerMode>;

  using ActionMask = std::bitset<kMaxNumActions>;
  using player_name_array_t = std::array<std::string, GameConstants::kNumPlayers>;

  using PolicyShape = Eigen::Sizes<kMaxNumActions>;
  using PolicyTensor = eigen_util::FTensor<PolicyShape>;
  using ValueTensor = GameResults::Tensor;
  using ValueShape = ValueTensor::Dimensions;
  using ActionValueShape = Eigen::Sizes<kMaxNumActions>;
  using ActionValueTensor = eigen_util::FTensor<ActionValueShape>;
  using ChanceEventShape = Eigen::Sizes<kMaxNumActions>;
  using ChanceDistribution = eigen_util::FTensor<ChanceEventShape>;

  using ValueArray = eigen_util::FArray<GameConstants::kNumPlayers>;
  using SymmetryMask = std::bitset<SymmetryGroup::kOrder>;
  using ActionSymmetryTable = core::ActionSymmetryTable<kMaxNumActions, SymmetryGroup>;
  using LocalPolicyArray = eigen_util::DArray<GameConstants::kMaxBranchingFactor>;
  using LocalActionValueArray = eigen_util::DArray<GameConstants::kMaxBranchingFactor>;

  static_assert(std::is_same_v<ValueArray, typename GameResults::ValueArray>);

  struct TrainingInfo {
    PolicyTensor* policy_target = nullptr;
    ActionValueTensor* action_values_target = nullptr;
    bool use_for_training = false;
  };

  /*
   * An ActionResponse is an action together with some optional auxiliary information:
   *
   * - victory_guarantee: whether the player believes their victory is guaranteed. GameServer can be
   *     configured to trust this guarantee, and immediately end the game. This can speed up
   *     simulations.
   *
   * - training_info: a pointer to a TrainingInfo object, to be used for NN training.
   */
  struct ActionResponse {
    ActionResponse(action_t a=-1) : action(a) {}

    action_t action = -1;
    bool victory_guarantee = false;
    TrainingInfo training_info;
  };

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
    core::action_mode_t action_mode;
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