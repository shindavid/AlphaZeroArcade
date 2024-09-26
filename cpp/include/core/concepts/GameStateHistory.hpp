#pragma once

#include <core/BasicTypes.hpp>

#include <concepts>

namespace core {
namespace concepts {

/*
 * GameStateHistory logically represents a sequence of recent game states. It is needed for 2
 * purposes:
 *
 * 1. Rules calculations: some games require a history of states to compute rules. For example, in
 *    chess, the threefold repetition rule and the fifty-move rule require a history of states.
 *
 * 2. Neural network input: the neural network input is constructed from an array of recent states.
 *    The GameStateHistory class must provide a way to access these states.
 *
 * Classes that implement the GameStateHistory concept only need to store enough recent states to
 * support these two use-cases. They do not need to store the entire game history.
 */
template <typename StateHistory, typename State, typename Rules>
concept GameStateHistory = requires(const State& const_state, const StateHistory& const_history,
                                    StateHistory& history, group::element_t sym) {

  /*
   * Clear the history.
   */
  { history.clear() };

  /*
   * Initialize the history with the given rules. We pass a Rules instance despite only needing
   * static methods to make the call-sites more readable (otherwise we would need to pass the Rules
   * as a template parameter, which would distastefully require the "template" keyword at the
   * call-site).
   */
  { history.initialize(Rules{}) };

  /*
   * Push back a copy of most recent state of the history, and return a reference to it.
   * Assumes that the history is not empty.
   */
  { history.extend() } -> std::same_as<State&>;

  /*
   * Push back the given state.
   */
  { history.update(const_state) };

  /*
   * Undo the most recent update() call. Assumes that the history is not empty, and that any two
   * undo() calls will have an update() call in between them.
   *
   * In general, classes implenting the GameStateHistory concept may require an extra slot of
   * storage to support this operation efficiently.
   *
   * This is used in a very specialized context within MCTS.
   */
  { history.undo() };

  /*
   * Return a reference to the most recent state in the history. Assumes that the history is not
   * empty.
   */
  { const_history.current() } -> std::same_as<const State&>;
  { history.current() } -> std::same_as<State&>;

  /*
   * begin()/end() methods to iterate over the part of the history needed for neural network input.
   */
  { history.begin() } -> std::same_as<decltype(history.end())>;
  { history.end() } -> std::same_as<decltype(history.begin())>;

  { const_history.begin() } -> std::same_as<decltype(const_history.end())>;
  { const_history.end() } -> std::same_as<decltype(const_history.begin())>;

  requires std::random_access_iterator<decltype(history.begin())>;
  requires requires(decltype(history.begin()) it) {
    { static_cast<State&>(*it) } -> std::convertible_to<State&>;
  };

  requires std::random_access_iterator<decltype(const_history.begin())>;
  requires requires(decltype(const_history.begin()) it) {
    { static_cast<const State&>(*it) } -> std::convertible_to<const State&>;
  };
};

}  // namespace concepts
}  // namespace core
