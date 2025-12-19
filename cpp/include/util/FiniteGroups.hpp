#pragma once

#include "util/CppUtil.hpp"

#include <concepts>
#include <cstdint>

namespace group {

using element_t = int32_t;
constexpr element_t kIdentity = 0;  // in every group, 0 is the identity element

namespace concepts {

template <typename G>
concept FiniteGroup = requires(group::element_t x, group::element_t y) {
  { util::decay_copy(G::kOrder) } -> std::same_as<int>;
  { G::inverse(x) } -> std::convertible_to<group::element_t>;

  // If the group acts on a set S, and s is a member of S, then:
  //
  // compose(x, y)(s) = x(y(s))
  { G::compose(x, y) } -> std::convertible_to<group::element_t>;
};

}  // namespace concepts

}  // namespace group

namespace groups {

struct TrivialGroup {
  static constexpr int kOrder = 1;
  static constexpr group::element_t inverse(group::element_t x) { return 0; }
  static constexpr group::element_t compose(group::element_t x, group::element_t y) { return 0; }
};

// Cn:
//
// 0: identity
// 1: clockwise rotation by 2pi/N
template <int N>
struct CyclicGroup {
  static constexpr int kOrder = N;
  static constexpr group::element_t inverse(group::element_t x);
  static constexpr group::element_t compose(group::element_t x, group::element_t y);
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
struct C2 : public CyclicGroup<2> {
  static constexpr group::element_t kIdentity = 0;
  static constexpr group::element_t kRot180 = 1;
};

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

}  // namespace groups

#include "inline/util/FiniteGroups.inl"
