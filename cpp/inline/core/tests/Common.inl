#include <core/tests/Common.hpp>

#include <core/BasicTypes.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/LoggingUtil.hpp>

#include <gtest/gtest.h>

#include <iostream>

namespace core {
namespace tests {

template <concepts::Game Game>
void Common<Game>::gtest_action_transforms() {
  using Constants = Game::Constants;
  using kNumActionsPerMode = Constants::kNumActionsPerMode;
  using PolicyTensor = Game::Types::PolicyTensor;
  for (action_mode_t action_mode = 0; action_mode < Game::Types::kNumActionModes; action_mode++) {
    int num_actions = util::get<kNumActionsPerMode>(action_mode);
    for (core::action_t action = 0; action < num_actions; ++action) {
      for (group::element_t sym = 0; sym < Game::SymmetryGroup::kOrder; ++sym) {
        core::action_t transformed_action = action;
        Game::Symmetries::apply(transformed_action, sym, action_mode);

        PolicyTensor policy;
        policy.setZero();
        policy(action) = 1;

        PolicyTensor transformed_policy = policy;
        Game::Symmetries::apply(transformed_policy, sym, action_mode);

        float sum = eigen_util::sum(transformed_policy);
        EXPECT_FLOAT_EQ(sum, 1);
        EXPECT_FLOAT_EQ(transformed_policy(transformed_action), 1);

        core::action_t inv_sym = Game::SymmetryGroup::inverse(sym);
        Game::Symmetries::apply(transformed_action, inv_sym, action_mode);
        EXPECT_EQ(transformed_action, action);
      }
    }
  }
}

}  // namespace tests
}  // namespace core
