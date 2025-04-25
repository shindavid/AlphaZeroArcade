#pragma once

#include <core/ActionSymmetryTable.hpp>
#include <core/BasicTypes.hpp>
#include <core/HibernationNotifier.hpp>
#include <core/concepts/GameConstants.hpp>
#include <core/concepts/GameResults.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>

#include <array>
#include <bitset>
#include <string>

#include <Eigen/Core>

namespace core {

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
struct GameTypes {
  using State = State_;
  using kNumActionsPerMode = GameConstants::kNumActionsPerMode;
  static constexpr int kNumActionModes = kNumActionsPerMode::size();
  static constexpr int kMaxNumActions = mp::MaxOf_v<kNumActionsPerMode>;

  using ActionMask = std::bitset<kMaxNumActions>;
  using player_name_array_t = std::array<std::string, GameConstants::kNumPlayers>;
  using player_bitset_t = std::bitset<GameConstants::kNumPlayers>;

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

  /*
   * Whenever use_for_training is true, policy_target and action_values_target should be non-null.
   *
   * The reverse, however, is not true: we can have use_for_training false, but have a non-null
   * policy_target. The reason for this is subtle: it's because we have an opponent-reply-policy
   * target. If we sample position 10 of the game, then we want to export the policy target for
   * position 11 (the opponent's reply), even if we don't sample position 11.
   */
  struct TrainingInfo {
    PolicyTensor* policy_target = nullptr;
    ActionValueTensor* action_values_target = nullptr;
    bool use_for_training = false;
  };

  struct ActionRequest {
    ActionRequest(const State& s, const ActionMask& va) : state(s), valid_actions(va) {}

    const State& state;
    const ActionMask& valid_actions;

    // If the player wishes to hibernate, it must later, when it's done with its asynchronous work,
    // call hibernation_notifier->notify() to notify the GameServer that the corresponding
    // GameSlot is ready to continue.
    HibernationNotifier* hibernation_notifier = nullptr;

    // If set to true, the player is being asked to play noisily, in order to add opening diversity.
    // Each player is free to interpret this in their own way.
    bool play_noisily = false;;
  };

  /*
   * An ActionResponse is an action together with some optional auxiliary information:
   *
   * - victory_guarantee: whether the player believes their victory is guaranteed. GameServer can be
   *     configured to trust this guarantee, and immediately end the game. This can speed up
   *     simulations.
   *
   * - training_info: used to generate targets for NN training.
   *
   * - yield_instruction: Indicates whether the player needs more time to think asynchronously. If
   *     set to a non-kYield value, then all other fields are ignored.
   *
   * - extra_enqueue_count: If set to a nonzero value, this instructs the GameServer to enqueue the
   *     current GameSlot this many additional times. This is useful for players that want to
   *     engage in multithreaded search. This should only be used for instruction type kYield.
   */
  struct ActionResponse {
    ActionResponse(action_t a = -1, core::yield_instruction_t y = core::kContinue)
        : action(a), yield_instruction(y) {}

    static ActionResponse yield() { return ActionResponse(-1, core::kYield); }
    static ActionResponse hibernate() { return ActionResponse(-1, core::kHibernate); }

    TrainingInfo training_info;
    action_t action = -1;
    int extra_enqueue_count = 0;
    core::yield_instruction_t yield_instruction = core::kContinue;
    bool victory_guarantee = false;
  };

  struct ChanceEventPreHandleResponse {
    ChanceEventPreHandleResponse(ActionValueTensor* a = nullptr,
                                 core::yield_instruction_t y = core::kContinue)
        : action_values(a), yield_instruction(y) {}

    static auto yield() { return ChanceEventPreHandleResponse(nullptr, core::kYield); }
    static auto hibernate() { return ChanceEventPreHandleResponse(nullptr, core::kHibernate); }

    ActionValueTensor* action_values = nullptr;
    core::yield_instruction_t yield_instruction = core::kContinue;
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
    const seat_index_t active_seat;
  };
};

}  // namespace core

#include <inline/core/GameTypes.inl>