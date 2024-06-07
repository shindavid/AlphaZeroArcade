#pragma once

#include <bitset>
#include <concepts>
#include <string>

#include <Eigen/Core>

#include <core/ActionOutcome.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

namespace core {

namespace concepts {

/*
 * All Game classes G must satisfy core::concepts::Game<G>.
 *
 * We use concepts rather than abstract classes primarily for efficiency reasons. Abstract classes
 * would require dynamic memory allocations and virtual method overhead. The dynamic memory aspect
 * would be particularly painful in the MCTS context, as variable-sized tensor calculations can be
 * quite a bit costlier than fixed-sized ones.
 */
template <class G>
concept Game = requires(
    const typename G::Position& const_pos) {
    // ,
    // State* _this, const State* const_this, typename State::Data* data,
    // const typename State::Data* const_data, typename State::Transform* transform,
    // typename State::PolicyTensor* policy, std::ostream& os) {

  { util::decay_copy(G::kNumPlayers) } -> std::same_as<int>;
  { util::decay_copy(G::kNumActions) } -> std::same_as<int>;
  { util::decay_copy(G::kMaxBranchingFactor) } -> std::same_as<int>;

  requires std::same_as<typename G::ActionMask, std::bitset<G::kNumActions>>;
  requires eigen_util::concepts::Shape<typename G::InputShape>;
  requires eigen_util::concepts::Shape<typename G::PolicyShape>;
  requires std::same_as<typename G::ValueArray, Eigen::Array<float, G::kNumPlayers, 1>>;
  requires std::same_as<typename G::ActionOutcome, core::ActionOutcome<typename G::ValueArray>>;

  requires std::is_default_constructible_v<typename G::Position>;
  requires std::is_trivially_copyable_v<typename G::Position>;

  requires util::concepts::Hashable<typename G::HashKey>;

  /*
   * Each Game class G must have:
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
  requires std::is_default_constructible_v<typename State::Data>;
  requires std::is_trivially_copyable_v<typename State::Data>;
  { const_this->data() } -> std::same_as<const typename State::Data&>;
  State{*const_data};
  State{};

  { util::decay_copy(State::kNumPlayers) } -> std::same_as<int>;
  { util::decay_copy(State::kMaxNumSymmetries) } -> std::same_as<int>;  // TODO: infer this from TransformList

  /*
   * Each GameState class must have an abstract inner class Transform that is used for symmetry
   * transformations, as well as an mp::TypeList TransformList that lists the derived Transform
   * classes.
   *
   * Each Transform class must have the following methods:
   *
   * - void apply(Data&)
   * - void apply(PolicyTensor&)
   * - void undo(Data&)
   * - void undo(PolicyTensor&)
   */
  typename State::Transform;
  requires mp::IsTypeListOf<typename State::TransformList, typename State::Transform>;
  { transform->apply(*data) } -> std::same_as<void>;
  { transform->apply(*policy) } -> std::same_as<void>;
  { transform->undo(*data) } -> std::same_as<void>;
  { transform->undo(*policy) } -> std::same_as<void>;

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
  requires eigen_util::concepts::Shape<typename State::ActionShape>;

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

  { _this->get_symmetry_indices() } -> std::same_as<std::bitset<State::kMaxNumSymmetries>>;

  /*
   * Return the current player.
   */
  { _this->get_current_player() } -> std::same_as<seat_index_t>;

  /*
   * Apply a given action to the state, and return a GameOutcome.
   */
  {
    _this->apply_move(typename GameStateTypes<State>::Action{})
  } -> std::same_as<typename GameStateTypes<State>::GameOutcome>;

  /*
   * Get the valid actions, as a bool tensor
   */
  { _this->get_valid_actions() } -> std::same_as<typename GameStateTypes<State>::ActionMask>;

  /*
   * A string representation of an action.
   */
  { State::action_to_str(typename GameStateTypes<State>::Action{}) } -> std::same_as<std::string>;

  /*
   * Game state evaluations are cached in a hash table. The GameState class must provide a
   * hash_key() method that returns a hashable type to serve as a key in this hash table.
   *
   * For most games, hash_key() can simply return *this. In a game like chess, where the state
   * stores a history of all past states to support the threefold-repetition rule, this would be a
   * poor choice, and simply returning data() would be more appropriate. On the other hand, in a
   * game like go, data() might not include the ko-state, since if the tensorizor includes recent
   * history, the ko-state can be inferred by the network from the history planes. But the ko-state
   * deserves to be part of the hash key.
   *
   * Due to potential game-specific nuances like this, we leave the details of hash_key() up to
   * each individual game.
   */
  typename State::HashKey;
  requires util::concepts::Hashable<typename State::HashKey>;
  { const_this->hash_key() } -> std::same_as<typename State::HashKey>;
};

}  // namespace concepts

}  // namespace core
