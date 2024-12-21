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
                             GameResultsTensor& results) {
  { GR::init_state(state) };
  { GR::get_legal_moves(const_history) } -> std::same_as<typename GameTypes::ActionMask>;
  { GR::get_action_mode(const_state) } -> std::same_as<core::action_mode_t>;
  { GR::get_current_player(const_state) } -> std::same_as<core::seat_index_t>;
  { GR::apply(history, core::action_t{}) };
  { GR::has_known_dist(state) };
  { GR::get_known_dist(state) } -> std::same_as<typename GameTypes::PolicyTensor>;

  // Return true iff the game has ended. If returning true, set results to the results of the game.
  { GR::is_terminal(const_state, last_player, last_action, results) } -> std::same_as<bool>;
};

}  // namespace concepts
}  // namespace core
