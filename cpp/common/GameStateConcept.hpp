#pragma once

#include <concepts>

#include <Eigen/Core>

#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/MctsResults.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

namespace common {

/*
 * All GameState classes must satisfy the GameStateConcept concept.
 */
template <class S>
concept GameStateConcept = requires(S state) {
  /*
   * The number of players in the game.
   */
  { util::decay_copy(S::kNumPlayers) } -> std::same_as<int>;

  /*
   * Return the total number of global actions in the game.
   *
   * For go, this is 19*19+1 = 362 (+1 because you can pass).
   * For connect-four, this is 7.
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
   */
  { util::decay_copy(S::kMaxNumLocalActions) } -> std::same_as<int>;

  /*
   * Return the current player.
   */
  { state.get_current_player() } -> std::same_as<player_index_t>;

  /*
   * Apply a given action to the state, and return a GameResult.
   */
  { state.apply_move(action_index_t()) } -> std::same_as<typename GameStateTypes_<S>::GameResult>;

  /*
   * Get the valid actions, as a util::BitSet.
   */
  { state.get_valid_actions() } -> is_bit_set_c;

  /*
   * A compact string representation, used for debugging purposes in conjunction with javascript visualizer.
   */
  { state.compact_repr() } -> std::same_as<std::string>;

  /*
   * Must be hashable.
   */
  { std::hash<S>{}(state) } -> std::convertible_to<std::size_t>;

  /*
   * For TUI-playing. Print a prompt to cout requesting an input, and then parse cin into an action.
   */
  { S::prompt_for_action() } -> std::same_as<common::action_index_t>;

  /*
   * Pretty-print neural network output to terminal for debugging purposes.
   */
  { S::xdump_nnet_output(MctsResults_<S>{}) };

  /*
   * Pretty-print mcts output to terminal for debugging purposes.
   *
   * TODO: clean up this interface
   */
  { S::xdump_mcts_output(
      typename GameStateTypes_<S>::ValueProbDistr{},
      typename GameStateTypes_<S>::LocalPolicyProbDistr{},
      MctsResults_<S>{})
  };
};

}  // namespace common
