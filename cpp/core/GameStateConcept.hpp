#pragma once

#include <concepts>

#include <Eigen/Core>

#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

namespace core {

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
   * The total number of actions in the game.
   *
   * In go, this is 362 = 19 * 19 + 1 (+1 for pass move)
   * Similarly, in othello, this is 65 = 8 * 8 + 1 (+1 for pass move)
   * In connect4, this is 7.
   *
   * Policy heads of neural networks will output arrays of this size.
   */
  { util::decay_copy(S::kNumGlobalActions) } -> std::same_as<int>;

  /*
   * For a given state s, let A(s) be the set of valid actions.
   *
   * kMaxNumLocalActions is an upper bound on the size of A(s) for all s.
   *
   * The main usage of this constant is that various parameters are by default set to some
   * formula based on this number. In such usages, this constant can be used as a proxy for the
   * branching factor of the game tree.
   *
   * The other usage is that a couple spots in the mcts code declare fixed-size data structures
   * based on this value. Declaring this value to be too small could thus lead to various
   * run-time errors. Too big could theoretically lead to performance inefficiencies, but this is
   * really minor; the bigger penalty for setting it too big is the aforementioned formula usage.
   *
   * In chess, this value can be as small as 218 (see: https://chess.stackexchange.com/a/8392).
   */
  { util::decay_copy(S::kMaxNumLocalActions) } -> std::same_as<int>;

  /*
   * Return the current player.
   */
  { state.get_current_player() } -> std::same_as<seat_index_t>;

  /*
   * Apply a given action to the state, and return a GameOutcome.
   */
  { state.apply_move(action_t()) } -> std::same_as<typename GameStateTypes<S>::GameOutcome>;

  /*
   * Get the valid actions, as a std::bitset
   */
  { state.get_valid_actions() } -> util::BitSetConcept;

  /*
   * Must be hashable (for use in MCGS).
   */
  { std::hash<S>{}(state) } -> std::convertible_to<std::size_t>;
};

}  // namespace core
