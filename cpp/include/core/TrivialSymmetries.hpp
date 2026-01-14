#pragma once

#include "core/BasicTypes.hpp"
#include "util/CompactBitSet.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

struct TrivialSymmetries {
  template <typename T>
  static util::CompactBitSet<1> get_mask(const T&) {
    util::CompactBitSet<1> mask;
    mask.set(0);
    return mask;
  }

  template <typename T>
  static void apply(T&, group::element_t) {}

  template <typename T>
  static void apply(T&, group::element_t, action_mode_t) {}
};

}  // namespace core
