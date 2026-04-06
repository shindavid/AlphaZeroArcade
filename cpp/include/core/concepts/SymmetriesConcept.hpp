#pragma once

#include "util/FiniteGroups.hpp"

#include <concepts>

namespace core::concepts {

template <typename S, typename Game, typename TensorEncodings, typename InputFrame>
concept Symmetries =
  requires(const InputFrame& const_frame, InputFrame& frame, group::element_t sym,
           typename TensorEncodings::PolicyEncoding::Tensor& policy,
           typename TensorEncodings::ActionValueEncoding::Tensor& action_values) {
    { S::get_mask(const_frame) } -> std::same_as<typename Game::Types::SymmetryMask>;
    { S::apply(frame, sym) };
    { S::apply(policy, sym, const_frame) };
    { S::apply(action_values, sym, const_frame) };
    { S::get_canonical_symmetry(const_frame) } -> std::same_as<group::element_t>;
  };

}  // namespace core::concepts
