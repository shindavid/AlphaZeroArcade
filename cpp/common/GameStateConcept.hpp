#pragma once

#include <concepts>

#include <Eigen/Core>

#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/MctsResults.hpp>
#include <util/CppUtil.hpp>

namespace common {

/*
 * All GameState classes must satisfy the GameStateConcept concept.
 *
 * We use concepts rather than abstract classes primarily for efficiency reasons. Abstract classes would require dynamic
 * memory allocations and virtual method overhead. The dynamic memory aspect would be particularly painful in the
 * MCTS context, as variable-sized tensor calculations can be quite a bit costlier than fixed-sized ones.
 */
template <class S>
concept GameStateConcept = requires(S state) {
  /*
   * The number of players in the game.
   */
  { util::decay_copy(S::kNumPlayers) } -> std::same_as<int>;

  /*
   * Return (an upper bound for) the total number of global actions in the game.
   *
   * For go, this is 19*19+1 = 362 (+1 because you can pass).
   * For connect-four, this is 7.
   *
   * Each MCTS node incurs a memory footprint cost of kNumGlobalAction bits, so it is worth keeping this number small.
   * However, it's not as important as minimizing kMaxNumLocalActions, which has 64x the footprint cost.
   */
  { util::decay_copy(S::kNumGlobalActions) } -> std::same_as<int>;

  /*
   * Return an upper bound on the number of local actions in the game.
   *
   * This can return the same value as get_num_global_actions(). Setting it to a smaller value allows for more
   * compact data structures, which should reduce overall memory footprint and improve performance.
   *
   * In a game like chess, this number can be much smaller than the global number, potentially as small as 218
   * (see: https://chess.stackexchange.com/a/8392).
   *
   * In the current implementation, each MCTS node contains an array of pointers of length kMaxNumLocalActions, which
   * corresponds to 64*kMaxNumLocalActions bits. It is worthwhile to keep this number small to reduce memory footprint.
   */
  { util::decay_copy(S::kMaxNumLocalActions) } -> std::same_as<int>;

  /*
   * Return the current player.
   */
  { state.get_current_player() } -> std::same_as<seat_index_t>;

  /*
   * Apply a given action to the state, and return a GameOutcome.
   */
  { state.apply_move(action_index_t()) } -> std::same_as<typename GameStateTypes<S>::GameOutcome>;

  /*
   * Get the valid actions, as a std::bitset
   */
  { state.get_valid_actions() } -> util::BitSetConcept;

  /*
   * Must be hashable (for use in MCGS).
   *
   * Currently, we still use MCTS, not MCGS, so this is a forward-looking requirement.
   */
  { std::hash<S>{}(state) } -> std::convertible_to<std::size_t>;

  /*
   * Pretty-print mcts output for debugging purposes.
   */
  { S::dump_mcts_output(
      typename GameStateTypes<S>::ValueProbDistr{},
      typename GameStateTypes<S>::LocalPolicyProbDistr{},
      MctsResults<S>{})
  };
};

}  // namespace common
