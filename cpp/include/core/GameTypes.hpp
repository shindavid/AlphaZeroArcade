#pragma once

#include "core/ActionSymmetryTable.hpp"
#include "core/BasicTypes.hpp"
#include "core/YieldManager.hpp"
#include "core/concepts/GameConstantsConcept.hpp"
#include "core/concepts/GameResultsConcept.hpp"
#include "util/CompactBitSet.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <Eigen/Core>

#include <array>
#include <string>

namespace core {

// TODO: some of the classes whose definitions are inlined here don't need to be. Move them out.
template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
struct GameTypes {
  using State = State_;
  using kNumActionsPerMode = GameConstants::kNumActionsPerMode;
  static constexpr int kNumActionModes = kNumActionsPerMode::size();
  static constexpr int kMaxNumActions = mp::MaxOf_v<kNumActionsPerMode>;
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

  using ValueArray = eigen_util::FArray<kNumPlayers>;
  using SymmetryMask = util::CompactBitSet<SymmetryGroup::kOrder>;
  using ActionSymmetryTable = core::ActionSymmetryTable<kMaxNumActions, SymmetryGroup>;
  using LocalPolicyArray = eigen_util::DArray<kMaxBranchingFactor>;
  using LocalPolicyArrayDouble = eigen_util::DArray<kMaxBranchingFactor, double>;
  using LocalActionValueArray =
    Eigen::Array<float, Eigen::Dynamic, kNumPlayers, Eigen::RowMajor, kMaxBranchingFactor>;

  static_assert(std::is_same_v<ValueArray, typename GameResults::ValueArray>);

  struct ChanceEventHandleRequest {
    ChanceEventHandleRequest(const YieldNotificationUnit& u, const State& s, action_t ca)
        : notification_unit(u), state(s), chance_action(ca) {}

    const YieldNotificationUnit& notification_unit;
    const State& state;
    action_t chance_action;
  };

  struct ActionRequest {
    ActionRequest(const State& s, const ActionMask& va, const YieldNotificationUnit& u,
                  node_aux_t a)
        : state(s), valid_actions(va), notification_unit(u), aux(a) {}

    ActionRequest(const State& s, const ActionMask& va) : state(s), valid_actions(va) {}

    const State& state;
    const ActionMask& valid_actions;
    YieldNotificationUnit notification_unit;

    // If set to true, the player is being asked to play noisily, in order to add opening diversity.
    // Each player is free to interpret this in their own way.
    bool play_noisily = false;
    node_aux_t aux = 0;
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
   *     set to a non-kContinue value, then the action/training_info/victory_guarantee fields are
   *     ignored. If set to kDrop, this indicates that this was an auxiliary thread launched for
   *     multithreaded search, and that the multithreaded part is over.
   *
   * - extra_enqueue_count: If set to a nonzero value, this instructs the GameServer to enqueue the
   *     current GameSlot this many additional times. This is useful for players that want to
   *     engage in multithreaded search. This should only be used for instruction type kYield.
   */
  struct ActionResponse {
    ActionResponse(action_t a = kNullAction, int e = 0,
                   core::yield_instruction_t y = core::kContinue)
        : action(a), extra_enqueue_count(e), yield_instruction(y) {}

    static ActionResponse yield(int e = 0) { return ActionResponse(kNullAction, e, core::kYield); }
    static ActionResponse drop() { return ActionResponse(kNullAction, 0, core::kDrop); }
    static ActionResponse resign() {
      ActionResponse r;
      r.resign_game = true;
      return r;
    }
    static ActionResponse backtrack(node_ix_t b) {
      ActionResponse r;
      r.backtrack_node_ix_ = b;
      return r;
    }
    template <typename T>
    void set_aux(T aux);

    bool is_aux_set() const { return aux_set_; }
    node_aux_t aux() const { return aux_; }
    node_ix_t backtrack_node_ix() const { return backtrack_node_ix_; }

    // TODO: make these private and add access methods
    action_t action = kNullAction;
    int extra_enqueue_count = 0;
    core::yield_instruction_t yield_instruction = core::kContinue;
    bool victory_guarantee = false;
    bool resign_game = false;  // If true, the player resigns the game.

   private:
    node_aux_t aux_ = 0;
    bool aux_set_ = false;
    node_ix_t backtrack_node_ix_ = kNullNodeIx;
  };
};

}  // namespace core

#include "inline/core/GameTypes.inl"
