#pragma once

#include "core/BasicTypes.hpp"
#include "core/RulesResult.hpp"

#include <concepts>

namespace core {
namespace concepts {

template <typename GR, typename GameTraits, typename State, typename Move>
concept GameRules = requires(const State& const_state, const State& prev_state, State& state,
                             const Move& move, seat_index_t seat) {
  { GR::init_state(state) };

  // Return true if the state is in a chance phase. In this case, the next change to the game state
  // will be determined by sampling from the chance distribution. Otherwise, the next change to the
  // game state will be determined by a player's action.
  { GR::is_chance_state(const_state) } -> std::same_as<bool>;

  // Assumes !is_chance_state(). Returns the seat index of the player whose turn it is to act.
  { GR::get_current_player(const_state) } -> std::same_as<core::seat_index_t>;

  // Assumes is_chance_state(). Returns a distribution to sample over.
  { GR::get_chance_distribution(const_state) };

  { GR::apply(state, move) };

  // Accepts a game state.
  //
  // Returns a RulesResult containing either the outcome of the game (if terminal)
  // or the set of legal moves (if action-phase), or a chance distribution (if chance-phase).
  { GR::analyze(const_state) } -> std::same_as<core::RulesResult<GameTraits>>;

  // Most classes can simply implement this as a call to the copy assignment operator. Others may
  // want to take advantage of the fact that other_states is a previous state in the same game.
  //
  // For example, for chess, the state stores a history of all past boards to support the
  // repetition rule. Resetting to a prior state can be implemented by simply truncating the
  // history, which is more efficient than copying the entire state.
  { GR::backtrack_state(state, prev_state) };
};

}  // namespace concepts
}  // namespace core
