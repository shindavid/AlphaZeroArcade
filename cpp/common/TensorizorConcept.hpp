#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>

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
  { util::decay_copy(T::kShape) } -> util::is_std_array_c;
};

}  // namespace common
