#pragma once

#include <core/BasicTypes.hpp>
#include <util/FiniteGroups.hpp>

#include <concepts>

namespace core {
namespace concepts {

template <typename GR, typename GameTypes, typename GameResultsTensor, typename State,
          typename StateHistory>
concept GameRules = requires(const State& const_state, const StateHistory& const_history,
                             State& state, StateHistory& history, group::element_t sym,
                             core::seat_index_t last_player, core::action_t last_action,
                             GameResultsTensor& results, action_mode_t action_mode) {
  // Initialize the state.
  { GR::init_state(state) };
  // Get the legal moves for the state history.
  { GR::get_legal_moves(const_history) } -> std::same_as<typename GameTypes::ActionMask>;
  // Get the action mode for the state.
  { GR::get_action_mode(const_state) } -> std::same_as<core::action_mode_t>;
  // Get the current player for the state.
  { GR::get_current_player(const_state) } -> std::same_as<core::seat_index_t>;
  // Apply the action to extend the state history.
  { GR::apply(history, core::action_t{}) };
  // TODO: make this function constexpr
  // Return true iff the action mode is chance mode.
  { GR::is_chance_mode(action_mode) } -> std::same_as<bool>;
  // Get the chance distribution for the state. Assumes the state is in chance mode.
  { GR::get_chance_distribution(const_state) } -> std::same_as<typename GameTypes::ChanceDistribution>;

  // Return true iff the game has ended. If returning true, set results to the results of the game.
  { GR::is_terminal(const_state, last_player, last_action, results) } -> std::same_as<bool>;
};

}  // namespace concepts
}  // namespace core
