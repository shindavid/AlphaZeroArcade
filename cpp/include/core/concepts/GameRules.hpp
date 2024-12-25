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
  { GR::init_state(state) };
  { GR::get_legal_moves(const_history) } -> std::same_as<typename GameTypes::ActionMask>;
  { GR::get_action_mode(const_state) } -> std::same_as<core::action_mode_t>;
  // Assumes the state is in player mode.
  { GR::get_current_player(const_state) } -> std::same_as<core::seat_index_t>;
  { GR::apply(history, core::action_t{}) };
  // TODO: make this function constexpr
  { GR::is_chance_mode(action_mode) } -> std::same_as<bool>;
  // Assumes the state is in chance mode.
  { GR::get_chance_distribution(const_state) } -> std::same_as<typename GameTypes::ChanceDistribution>;

  // Return true iff the game has ended. If returning true, set results to the results of the game.
  // last_player is allowed to be -1 if the last_player was a chance node.
  // TODO: pass in last_mode
  { GR::is_terminal(const_state, last_player, last_action, results) } -> std::same_as<bool>;
};

}  // namespace concepts
}  // namespace core
