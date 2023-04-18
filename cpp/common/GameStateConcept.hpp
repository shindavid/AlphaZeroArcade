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
   * The serialize_* and deserialize_* methods are used primarily to communicate states and actions between different
   * processes (like a game server with a remote game client). The expected declaration of these methods would look like
   * this:
   *
   * static size_t serialize_action(char* buf, size_t buf_size, action_index_t action);
   * static void deserialize_action(const char* buf, action_index_t* action);
   *
   * size_t serialize_action_prompt(char* buffer, size_t buffer_size, const ActionMask& valid_actions) const { return 0; }
   * void deserialize_action_prompt(const char* buffer, ActionMask* valid_actions) const {}
   *
   * size_t serialize_state_change(char* buf, size_t buf_size, seat_index_t seat, action_index_t action) const;
   * void deserialize_state_change(const char* buf, seat_index_t* seat, action_index_t* action);
   *
   * size_t serialize_game_end(char* buffer, size_t buffer_size, const GameOutcome& outcome) const;
   * void deserialize_game_end(const char* buffer, GameOutcome* outcome);
   *
   * The serialize_* methods are expected to write to the given buffer, and return the number of bytes written.
   *
   * The deserialize_* methods are expected to act as inverse-functions of their serialize_* counterparts, writing to
   * the given pointers.
   *
   * The serialize_state_change() method is expected to be called on the state *after* the action was performed by
   * player (resulting in outcome).
   *
   * The deserialize_state_change() method is expected to be called on the state *before* the change, and is expected
   * to perform the change on this, while populating the player/action/outcome pointers with the relevant information
   * about the state change.
   *
   * It is best for the string representations to be as stable as possible, to maximize chances of forward-compatibility
   * with future versions of the code. For example, for deterministic games like chess/go/Connect4, the string
   * representing the state change only needs to contain the action taken. The player and outcome can be derived from
   * the previous state and the action. This is superior to something like the byte-representation of the state, since
   * that could change between versions of the code, leading to backwards-incompatibility.
   */
  { S::serialize_action(std::declval<char*>(), size_t(), action_index_t()) } -> std::same_as<size_t>;
  { S::deserialize_action(std::declval<char*>(), std::declval<action_index_t*>()) };

  { state.serialize_action_prompt(std::declval<char*>(), size_t(), typename GameStateTypes<S>::ActionMask{}) } -> std::same_as<size_t>;
  { state.deserialize_action_prompt(std::declval<char*>(), std::declval<typename GameStateTypes<S>::ActionMask*>()) };

  { state.serialize_state_change(std::declval<char*>(), size_t(), seat_index_t(), action_index_t()) } -> std::same_as<size_t>;
  { state.deserialize_state_change(std::declval<char *>(), std::declval<seat_index_t *>(), std::declval<action_index_t *>()) };

  { state.serialize_game_end(std::declval<char*>(), size_t(), typename GameStateTypes<S>::GameOutcome{}) } -> std::same_as<size_t>;
  { state.deserialize_game_end(std::declval<char *>(), std::declval<typename GameStateTypes<S>::GameOutcome*>()) };

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
   * Pretty-print mcts output to terminal for debugging purposes.
   *
   * TODO: clean up this interface
   */
  { S::dump_mcts_output(
      typename GameStateTypes<S>::ValueProbDistr{},
      typename GameStateTypes<S>::LocalPolicyProbDistr{},
      MctsResults<S>{})
  };
};

}  // namespace common
