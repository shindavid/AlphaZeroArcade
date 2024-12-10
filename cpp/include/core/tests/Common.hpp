#pragma once

#include <core/concepts/Game.hpp>

/*
 * This file contains unit-testing code that can be shared by all games.
 */

namespace core {
namespace tests {

template <concepts::Game Game>
struct Common {
  using Policy = Game::Types::Policy;

  /*
   * For every possible (action, sym) pair:
   *
   *  1. Construct a policy tensor with a 1 at the action index.
   *  2. Apply the symmetry to the action and to the tensor.
   *  3. Check that the action and the tensor are consistent.
   *  4. Check that the inverse symmetry acts as expected on the action and the tensor.
   *
   * If these checks pass, returns true. Else, prints failure info to stdout, and returns false.
   */
  static bool test_action_transforms(const char* func);
  static void gtest_action_transforms();

  template<action_type_t ActionType=0>
  static bool policies_match(const Policy& p1, const Policy& p2) {
    return eigen_util::equal(std::get<ActionType>(p1), std::get<ActionType>(p2));
  }
};

}  // namespace tests
}  // namespace core

#include <inline/core/tests/Common.inl>
