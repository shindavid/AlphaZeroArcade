#pragma once

#include <concepts>
#include <cstdint>
#include <initializer_list>

#include <util/CppUtil.hpp>

namespace common {

/*
 * All Tensorizor classes must satisfy the TensorizorConcept concept.
 */
template <class T>
concept TensorizorConcept = requires(T tensorizor) {
  /*
   * The number of players in the game.
   */
  { util::decay_copy(T::kShape) } -> std::same_as<std::initializer_list<size_t>>;
};

}  // namespace common
