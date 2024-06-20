#pragma once

#include <core/BasicTypes.hpp>

#include <concepts>

namespace core {
namespace concepts {

template <typename GR, typename GameTypes, typename BaseState, typename FullState>
concept GameRules = requires(const BaseState& const_base_state, const FullState& const_full_state,
                             FullState& full_state) {
  { GR::get_legal_moves(const_full_state) } -> std::same_as<typename GameTypes::ActionMask>;
  { GR::get_current_player(const_base_state) } -> std::same_as<core::seat_index_t>;
  { GR::apply(full_state, core::action_t{}) } -> std::same_as<typename GameTypes::ActionOutcome>;
  { GR::get_symmetries(const_full_state) } -> std::same_as<typename GameTypes::SymmetryIndexSet>;
};

}  // namespace concepts
}  // namespace core
