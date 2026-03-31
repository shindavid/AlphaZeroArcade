#pragma once

#include "core/BasicTypes.hpp"
#include "util/FiniteGroups.hpp"

#include <concepts>

namespace core::concepts {

template <typename S, typename Game, typename InputFrame>
concept Symmetries =
  requires(const InputFrame& const_frame, InputFrame& frame, group::element_t sym,
           core::game_phase_t game_phase, typename Game::Types::PolicyTensor& policy,
           typename Game::Types::ActionValueTensor& action_values) {
    { S::get_mask(const_frame) } -> std::same_as<typename Game::Types::SymmetryMask>;
    { S::apply(frame, sym) };
    { S::apply(policy, sym, game_phase) };
    { S::apply(action_values, sym, game_phase) };
    { S::get_canonical_symmetry(const_frame) } -> std::same_as<group::element_t>;
  };

}  // namespace core::concepts
