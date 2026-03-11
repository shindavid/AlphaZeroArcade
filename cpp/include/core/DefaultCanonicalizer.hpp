#pragma once

#include "util/FiniteGroups.hpp"

#include <concepts>

namespace core {

/*
 * The DefaultCanonicalizer applies every possible symmetry to an input frame, sorts the results,
 * and returns the first one.
 *
 * This can be inefficient, but it is guaranteed to work for any game, as long as the game's
 * InputFrame class is a comparable type.
 */
template <typename InputFrame, typename Symmetries>
class DefaultCanonicalizer {
 public:
  static_assert(std::totally_ordered<InputFrame>, "InputFrame must be totally ordered");

  static group::element_t get(const InputFrame& frame);
};

}  // namespace core

#include "inline/core/DefaultCanonicalizer.inl"
