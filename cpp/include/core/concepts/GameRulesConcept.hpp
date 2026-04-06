#pragma once

#include "core/BasicTypes.hpp"
#include "core/RulesResult.hpp"

#include <concepts>

namespace core {
namespace concepts {

template <typename GR, typename GameTypes, typename State, typename Move>
concept GameRules = requires(const State& const_state, const State& prev_state, State& state,
                             const Move& move, game_phase_t game_phase) {
  { GR::init_state(state) };
  { GR::get_game_phase(const_state) } -> std::same_as<core::game_phase_t>;

  // Assumes the state is in an action phase
  { GR::get_current_player(const_state) } -> std::same_as<core::seat_index_t>;

  { GR::is_chance_phase(game_phase) } -> std::same_as<bool>;
  { GR::apply(state, move) };

  // Accepts a game state.
  //
  // Returns a RulesResult containing either the outcome of the game (if terminal)
  // or the set of legal moves (if action-phase), or a chance distribution (if chance-phase).
  { GR::analyze(const_state) } -> std::same_as<core::RulesResult<GameTypes>>;

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
