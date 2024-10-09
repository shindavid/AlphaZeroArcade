#include <util/FiniteGroups.hpp>

#include <util/Asserts.hpp>
#include <util/Random.hpp>

namespace groups {

template <int N>
constexpr group::element_t CyclicGroup<N>::inverse(group::element_t x) {
  return N - x;
}

template <int N>
constexpr group::element_t CyclicGroup<N>::compose(group::element_t x, group::element_t y) {
  return (x + y) % N;
}

template <int N>
constexpr group::element_t DihedralGroup<N>::inverse(group::element_t x) {
  if (x < N) {  // Rotations
    return (N - x) % N;
  } else {  // Reflections
    return x;
  }
}

template <int N>
constexpr group::element_t DihedralGroup<N>::compose(group::element_t x, group::element_t y) {
  if (x < N && y < N) {  // Rotation * Rotation
    return (x + y) % N;
  } else if (x < N && y >= N) {  // Rotation * Reflection
    return (y + x) % N + N;
  } else if (x >= N && y < N) {  // Reflection * Rotation
    return (x - y + N) % N + N;
  } else {  // Reflection * Reflection
    return (x - y + N) % N;
  }
}

}  // namespace groups
