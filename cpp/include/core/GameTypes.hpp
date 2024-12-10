#pragma once

#include <core/ActionSymmetryTable.hpp>
#include <core/BasicTypes.hpp>
#include <core/concepts/GameConstants.hpp>
#include <core/concepts/GameResults.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/IndexedDispatcher.hpp>
#include <util/MetaProgramming.hpp>

#include <array>
#include <bitset>
#include <string>
#include <variant>

#include <Eigen/Core>

namespace core {

template <concepts::GameConstants GameConstants, typename State, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
struct GameTypes {
  static constexpr int kNumActionTypes = GameConstants::kNumActionsPerType::size();

  using kNumActionsPerType = GameConstants::kNumActionsPerType;
  using ActionTypeDispatcher = util::IndexedDispatcher<kNumActionTypes>;
  using ActionMaskVariant = mp::TransformIntSequence_t<std::variant, kNumActionsPerType, std::bitset>;
  using player_name_array_t = std::array<std::string, GameConstants::kNumPlayers>;

  using PolicyShapeVariant =
      mp::TransformIntSequence_t<std::variant, kNumActionsPerType, eigen_util::make_1d_shape_t>;
  using PolicyTensorVariant = mp::Transform_t<std::variant, PolicyShapeVariant, eigen_util::FTensor>;
  using ValueTensor = GameResults::Tensor;
  using ValueShape = ValueTensor::Dimensions;
  using ActionValueShapeVariant =
      mp::TransformIntSequence_t<std::variant, kNumActionsPerType, eigen_util::make_1d_shape_t>;
  using ActionValueTensorVariant = mp::Transform_t<std::variant, PolicyShapeVariant, eigen_util::FTensor>;

  using ValueArray = eigen_util::FArray<GameConstants::kNumPlayers>;
  using SymmetryMask = std::bitset<SymmetryGroup::kOrder>;
  using ActionSymmetryTable = core::ActionSymmetryTable<GameConstants, SymmetryGroup>;
  using LocalPolicyArray = eigen_util::DArray<GameConstants::kMaxBranchingFactor>;
  using LocalActionValueArray = eigen_util::DArray<GameConstants::kMaxBranchingFactor>;

  static_assert(std::is_same_v<ValueArray, typename GameResults::ValueArray>);

  struct TrainingInfo {
    PolicyTensorVariant* policy_target = nullptr;
    ActionValueTensorVariant* action_values_target = nullptr;
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
    ActionMaskVariant valid_actions;
    PolicyTensorVariant counts;
    PolicyTensorVariant policy_target;
    PolicyTensorVariant policy_prior;
    PolicyTensorVariant Q;
    PolicyTensorVariant Q_sq;
    ActionValueTensorVariant action_values;
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
    const PolicyTensorVariant* policy;
    const PolicyTensorVariant* next_policy;
    const ActionValueTensorVariant* action_values;
  };
};

}  // namespace core

#include <inline/core/GameTypes.inl>