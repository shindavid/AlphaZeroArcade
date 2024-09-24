#pragma once

#include <core/BasicTypes.hpp>
#include <util/FiniteGroups.hpp>

#include <concepts>

namespace core {
namespace concepts {

template <typename GR, typename GameTypes, typename BaseState, typename StateHistory>
concept GameRules = requires(const BaseState& const_base_state, const StateHistory& const_history,
                             BaseState& base_state, StateHistory& history, group::element_t sym) {
  { GR::init_state(base_state) };
  { GR::init_state(base_state, sym) };
  { GR::get_legal_moves(const_history) } -> std::same_as<typename GameTypes::ActionMask>;
  { GR::get_current_player(const_base_state) } -> std::same_as<core::seat_index_t>;
  { GR::apply(history, core::action_t{}) } -> std::same_as<typename GameTypes::ActionOutcome>;
};

}  // namespace concepts
}  // namespace core
