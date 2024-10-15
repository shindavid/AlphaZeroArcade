#include <core/tests/Common.hpp>

#include <core/BasicTypes.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>

#include <gtest/gtest.h>

#include <iostream>

namespace core {
namespace tests {

template <concepts::Game Game>
bool Common<Game>::test_action_transforms(const char* func) {
  using PolicyTensor = Game::Types::PolicyTensor;
  for (core::action_t action = 0; action < Game::Constants::kNumActions; ++action) {
    for (group::element_t sym = 0; sym < Game::SymmetryGroup::kOrder; ++sym) {
      core::action_t transformed_action = action;
      Game::Symmetries::apply(transformed_action, sym);

      PolicyTensor policy;
      policy.setZero();
      policy(action) = 1;

      PolicyTensor transformed_policy = policy;
      Game::Symmetries::apply(transformed_policy, sym);

      float sum = eigen_util::sum(transformed_policy);
      if (sum != 1) {
        printf("Failure in %s() at %s:%d\n", func, __FILE__, __LINE__);
        printf("sym=%d action:%d->%d\n", sym, action, transformed_action);
        printf("Unexpected sum(transformed_policy): %.f\n", sum);
        return false;
      }

      if (transformed_policy(transformed_action) != 1) {
        printf("Failure in %s() at %s:%d\n", func, __FILE__, __LINE__);
        printf("sym=%d action:%d->%d\n", sym, action, transformed_action);
        printf("Not consistent with transformed policy:\n");
        for (int i = 0; i < Game::Constants::kNumActions; ++i) {
          printf("%s: %.f\n", Game::IO::action_to_str(i).c_str(), transformed_policy(i));
        }
        return false;
      }

      core::action_t inv_sym = Game::SymmetryGroup::inverse(sym);
      Game::Symmetries::apply(transformed_action, inv_sym);
      if (transformed_action != action) {
        printf("Failure in %s() at %s:%d\n", func, __FILE__, __LINE__);
        printf("With sym=%d, expected transformed_action=%d, but got %d\n", sym, action,
               transformed_action);
        return false;
      }
    }
  }

  return true;
}

template <concepts::Game Game>
void Common<Game>::gtest_action_transforms() {
  using PolicyTensor = Game::Types::PolicyTensor;
  for (core::action_t action = 0; action < Game::Constants::kNumActions; ++action) {
    for (group::element_t sym = 0; sym < Game::SymmetryGroup::kOrder; ++sym) {
      core::action_t transformed_action = action;
      Game::Symmetries::apply(transformed_action, sym);

      PolicyTensor policy;
      policy.setZero();
      policy(action) = 1;

      PolicyTensor transformed_policy = policy;
      Game::Symmetries::apply(transformed_policy, sym);

      float sum = eigen_util::sum(transformed_policy);
      EXPECT_FLOAT_EQ(sum, 1);
      EXPECT_FLOAT_EQ(transformed_policy(transformed_action), 1);

      core::action_t inv_sym = Game::SymmetryGroup::inverse(sym);
      Game::Symmetries::apply(transformed_action, inv_sym);
      EXPECT_EQ(transformed_action, action);
    }
  }
}

}  // namespace tests
}  // namespace core
