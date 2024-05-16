#pragma once

#include <concepts>
#include <string>

#include <Eigen/Core>

#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

namespace core {

/*
 * All GameState classes must satisfy the GameStateConcept concept.
 *
 * We use concepts rather than abstract classes primarily for efficiency reasons. Abstract classes
 * would require dynamic memory allocations and virtual method overhead. The dynamic memory aspect
 * would be particularly painful in the MCTS context, as variable-sized tensor calculations can be
 * quite a bit costlier than fixed-sized ones.
 */
template <class State>
concept GameStateConcept = requires(State state, const State& const_state,
                                    const typename State::Data& data, std::ostream& os) {
  /*
   * Each GameState class must have:
   *
   * - A trivially-copyable inner class called Data, containing data describing the game state.
   * - A constructor that accepts a Data object as an argument.
   * - A data() method that returns a reference to a Data object.
   *
   * When serializing a GameState, only the Data returned by data() is serialized. Deserialization
   * is accomplished by calling the constructor with the deserialized Data object, along with
   * optionally passing past GameState instances.
   *
   * For most games, GameState does not need any additional data members besides a Data instance,
   * and past GameState instances are not needed.
   *
   * An example of a game where you may want this additional machinery is chess. In chess, in order
   * to determine whether a game is drawn due to the threefold repetition rule, the game state
   * technically needs to store all previous game states in a dynamic data structure. Serializing
   * and deserializing this data structure would be cumbersome. It is better to just store the board
   * state, and to reconstruct the data structure from past board states if needed.
   */
  typename State::Data;
  requires std::is_trivially_copyable_v<typename State::Data>;
  { const_state.data() } -> std::same_as<const typename State::Data&>;
  State{data};
  State{};

  /*
   * The number of players in the game.
   */
  { util::decay_copy(State::kNumPlayers) } -> std::same_as<int>;

  /*
   * The maximum number of symmetries.
   */
  { util::decay_copy(State::kMaxNumSymmetries) } -> std::same_as<int>;

  { state.get_symmetry_indices() } -> std::same_as<std::bitset<State::kMaxNumSymmetries>>;

  /*
   * The string that will be used as a delimiter to separate a sequence of action strings,
   * as returned by State::action_to_str().
   */
  { State::action_delimiter() } -> std::same_as<std::string>;

  /*
   * The shape of the tensor used to represent an action.
   *
   * For many games, the best choice is a 1D tensor, whose single dimension corresponds to the
   * number of valid actions in the game.
   *
   * In some games, it is more convenient to use a multidimensional tensor. For example, in chess,
   * AlphaZero represents an action as an (8, 8, 73) tensor. The (8, 8) corresponds to the
   * starting position of the piece and the 73 corresponds to various move-types (including
   * pawn promotions).
   */
  { typename State::ActionShape{} } -> eigen_util::ShapeConcept;

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
   *
   * TODO: decouple the formula usage (which probably wants instead kTypicalBranchingFactor) from
   * the structure sizing usage (which needs kMax*).
   */
  { util::decay_copy(State::kMaxNumLocalActions) } -> std::same_as<int>;

  /*
   * Return the current player.
   */
  { state.get_current_player() } -> std::same_as<seat_index_t>;

  /*
   * Apply a given action to the state, and return a GameOutcome.
   */
  {
    state.apply_move(typename GameStateTypes<State>::Action{})
  } -> std::same_as<typename GameStateTypes<State>::GameOutcome>;

  /*
   * Get the valid actions, as a bool tensor
   */
  { state.get_valid_actions() } -> std::same_as<typename GameStateTypes<State>::ActionMask>;

  /*
   * A string representation of an action.
   */
  { State::action_to_str(typename GameStateTypes<State>::Action{}) } -> std::same_as<std::string>;

  /*
   * Must be hashable (for use in MCGS).
   */
  { std::hash<State>{}(state) } -> std::convertible_to<std::size_t>;
};

}  // namespace core
