#include <util/FiniteGroups.hpp>

#include <util/Asserts.hpp>
#include <util/Random.hpp>

namespace groups {

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

template <concepts::FiniteGroupList L> group::element_t get_random_element(int index) {
  group::element_t x = -1;
  constexpr int length = mp::Length_v<L>;

  mp::constexpr_for<0, length, 1>([&](auto a) {
    using Group = mp::TypeAt_t<L, a>;
    if (a == index) {
      x = util::Random::uniform_sample(0, Group::kOrder);
    }
  });
  util::release_assert(x != -1, "index %d out of range [0, %d)", index, length);
  return x;
}

template <concepts::FiniteGroupList L>
group::element_t get_inverse_element(int index, group::element_t x) {
  group::element_t y = -1;
  constexpr int length = mp::Length_v<L>;

  mp::constexpr_for<0, length, 1>([&](auto a) {
    using Group = mp::TypeAt_t<L, a>;
    if (a == index) {
      y = Group::inverse(x);
    }
  });
  util::release_assert(y != -1, "index %d out of range [0, %d)", index, length);
  return y;
}

}  // namespace groups
