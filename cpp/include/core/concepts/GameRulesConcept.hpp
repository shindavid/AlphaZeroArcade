#pragma once

#include "core/BasicTypes.hpp"
#include "util/FiniteGroups.hpp"

#include <concepts>

namespace core {
namespace concepts {

template <typename GR, typename GameTypes, typename GameResultsTensor, typename State>
concept GameRules =
  requires(const State& const_state, const State& prev_state, State& state, group::element_t sym,
           core::seat_index_t last_active_seat, core::action_t last_action,
           GameResultsTensor& results, action_mode_t action_mode) {
  { GR::init_state(state) };
  { GR::get_legal_moves(const_state) } -> std::same_as<typename GameTypes::ActionMask>;
  { GR::get_action_mode(const_state) } -> std::same_as<core::action_mode_t>;

  // Assumes the state is in player mode.
  { GR::get_current_player(const_state) } -> std::same_as<core::seat_index_t>;
  { GR::apply(state, core::action_t{}) };

  // TODO: make this function constexpr
  { GR::is_chance_mode(action_mode) } -> std::same_as<bool>;

  // Assumes the state is in chance mode.
  {
    GR::get_chance_distribution(const_state)
  } -> std::same_as<typename GameTypes::ChanceDistribution>;

  // Return true iff the game has ended. If returning true, set results to the results of the
  // game. last_action is the last action that was taken (whether by a player or a chance-event),
  // and last_active_seat is the seat that was active when that action was taken. For player
  // events, last_active_seat will be the seat of the player who took the action. For chance
  // events, last_active_seat will be the seat of the player who was active before the chance
  // event.
  { GR::is_terminal(const_state, last_active_seat, last_action, results) } -> std::same_as<bool>;

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
