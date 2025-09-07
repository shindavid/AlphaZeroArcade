#pragma once

#include "core/concepts/GameConcept.hpp"

/*
 * This file contains unit-testing code that can be shared by all games.
 */

namespace core {
namespace tests {

template <concepts::Game Game>
struct Common {
  /*
   * For every possible (action, sym) pair:
   *
   *  1. Construct a policy tensor with a 1 at the action index.
   *  2. Apply the symmetry to the action and to the tensor.
   *  3. Check that the action and the tensor are consistent.
   *  4. Check that the inverse symmetry acts as expected on the action and the tensor.
   */
  static void gtest_action_transforms();
};

// Dispatches to standard gtest main function, while adding LoggingUtil cmdline params
int main(int argc, char** argv);

}  // namespace tests
}  // namespace core

#include "inline/core/tests/Common.inl"
