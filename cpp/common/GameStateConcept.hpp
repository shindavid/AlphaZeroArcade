#pragma once

#include <concepts>

#include <Eigen/Core>

#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/MctsResults.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

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
   * The shape of the tensor representation of an action policy.
   *
   * For go, this is simply (362) - the possibility of a pass move makes it awkward to use a (19, 19) tensor, so we
   * simply flatten the action space into a single dimension (19 * 19 + 1 = 362).
   *
   * For chess, a better representation is possible: (8, 8, 73). See the AlphaZero paper for details.
   *
   * In principle, this belongs to the Tensorizor, not the GameState. However, we represent each action as an int that
   * corresponds to an index in the flattened (row-major) policy tensor, and the GameState interface is tied to a
   * specific action->int mapping. That warrants putting this here. A true separation would demand an Action class,
   * with the action->int specified by the Tensorizor. This separation entails a performance cost, which is not
   * justified.
   */
  { typename S::PolicyShape{} } -> eigen_util::ShapeConcept;

  /*
   * The number of players in the game.
   */
  { util::decay_copy(S::kNumPlayers) } -> std::same_as<int>;

  /*
   * For a given state s, let A(s) be the set of valid actions.
   *
   * kMaxNumLocalActions is an upper bound on the size of A(s) for all s.
   *
   * It is not necessary to set this as tight as possible, but tighter is better. Setting it to a smaller value allows
   * for more compact data structures, which should reduce overall memory footprint and improve performance.
   *
   * In chess, this value can be as small as 218 (see: https://chess.stackexchange.com/a/8392).
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
