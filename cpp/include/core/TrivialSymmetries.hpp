#pragma once

#include "core/BasicTypes.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

struct TrivialSymmetries {
  template <typename T>
  static std::bitset<1> get_mask(const T&) {
    return std::bitset<1>(1);
  }

  template <typename T>
  static void apply(T&, group::element_t) {}

  template <typename T>
  static void apply(T&, group::element_t, action_mode_t) {}

  template <typename T>
  static group::element_t get_canonical_symmetry(const T&) {
    return group::kIdentity;
  }
};

}  // namespace core
