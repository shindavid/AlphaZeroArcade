#pragma once

#include <concepts>
#include <cstdint>
#include <initializer_list>

template <class T>
std::decay_t<T> decay_copy(T&&);

namespace common {

/*
 * All Tensorizor classes must satisfy the TensorizorConcept concept.
 */
template <class T>
concept TensorizorConcept = requires(T tensorizor) {
  /*
   * The number of players in the game.
   */
  { decay_copy(T::kShape) } -> std::same_as<std::initializer_list<size_t>>;
};

}  // namespace common
