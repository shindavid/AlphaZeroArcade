#pragma once

#include <concepts>

#include <common/GameResult.hpp>
#include <common/Types.hpp>
#include <util/BitSet.hpp>

namespace common {

/*
 * All GameState classes must satisfy the GameStateConcept concept.
 */
template <class S>
concept GameStateConcept = requires(S state) {
  /*
   * The number of players in the game.
   */
  { S::get_num_players() } -> std::same_as<int>;

  /*
   * Return the total number of global actions in the game.
   *
   * For go, this is 19*19+1 = 362 (+1 because you can pass).
   * For connect-four, this is 7.
   */
  { S::get_num_global_actions() } -> std::same_as<int>;

  /*
   * Return an upper bound on the number of local actions in the game.
   *
   * This can return the same value as get_num_global_actions(). Setting it to a smaller value allows for more
   * compact data structures, which should reduce overall memory footprint and improve performance.
   *
   * In a game like chess, this number can be much smaller than the global number, potentially as small as 218
   * (see: https://chess.stackexchange.com/a/8392).
   */
  { S::get_max_num_local_actions() } -> std::same_as<int>;

  /*
   * Return the current player.
   */
  { state.get_current_player() } -> std::same_as<player_index_t>;

  /*
   * Apply a given action to the state, and return a GameResult.
   */
  { state.apply_move(action_index_t()) } -> is_game_result_c;

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
};

}  // namespace common
