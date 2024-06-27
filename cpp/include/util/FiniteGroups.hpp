#pragma once

#include <util/CppUtil.hpp>
#include <util/MetaProgramming.hpp>

#include <concepts>
#include <cstdint>

namespace group {

using element_t = int32_t;
constexpr element_t kIdentity = 0;  // in every group, 0 is the identity element

}  // namespace group

namespace concepts {

template <typename G>
concept FiniteGroup = requires(group::element_t x, group::element_t y) {
  { util::decay_copy(G::kOrder) } -> std::same_as<int>;
  { G::inverse(x) } -> std::convertible_to<int>;
  { G::compose(x, y) } -> std::convertible_to<int>;
};

}  // namespace concepts

namespace groups {

struct TrivialGroup {
  static constexpr int kOrder = 1;
  static constexpr group::element_t inverse(group::element_t x) { return 0; }
  static constexpr group::element_t compose(group::element_t x, group::element_t y) { return 0; }
};

// Dn:
//
// 0: identity
// 1: clockwise rotation by 2pi/N
// N: reflection about x-axis
template <int N>
struct DihedralGroup {
  static constexpr int kOrder = 2 * N;
  static constexpr group::element_t inverse(group::element_t x);
  static constexpr group::element_t compose(group::element_t x, group::element_t y);
};

/*
 * Specialization to provide named elements
 */
struct D1 : public DihedralGroup<1> {
  static constexpr group::element_t kIdentity = 0;
  static constexpr group::element_t kFlip = 1;
};

/*
 * Specialization to provide named elements
 */
struct D4 : public DihedralGroup<4> {
  static constexpr group::element_t kIdentity = 0;
  static constexpr group::element_t kRot90 = 1;
  static constexpr group::element_t kRot180 = 2;
  static constexpr group::element_t kRot270 = 3;
  static constexpr group::element_t kFlipVertical = 4;
  static constexpr group::element_t kFlipMainDiag = 5;  // top-left to bot-right
  static constexpr group::element_t kMirrorHorizontal = 6;
  static constexpr group::element_t kFlipAntiDiag = 7;  // top-right to bot-left
};

template <typename G>
struct FiniteGroupPredicate {
  static constexpr bool value = concepts::FiniteGroup<G>;
};

}  // namespace groups

namespace concepts {

template <typename T>
concept FiniteGroupList = mp::IsTypeListSatisfying<T, groups::FiniteGroupPredicate>;

}  // namespace concepts

namespace groups {

/*
 * groups::get_random_element<L>(k) returns a random element of the k'th Group of L.
 */
template <concepts::FiniteGroupList L> group::element_t get_random_element(int index);

/*
 * groups::get_inverse_element<L>(k, x) returns the inverse of x in the k'th Group of L.
 */
template <concepts::FiniteGroupList L>
group::element_t get_inverse_element(int index, group::element_t x);

}  // namespace groups

#include <inline/util/FiniteGroups.inl>
