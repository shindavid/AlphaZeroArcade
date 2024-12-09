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
  using Constants = Game::Constants;
  using kNumActionsPerType = Constants::kNumActionsPerType;
  using ActionTypeDispatcher = Game::Types::ActionTypeDispatcher;
  using PolicyTensorVariant = Game::Types::PolicyTensorVariant;

  for (action_type_t action_type = 0; action_type < Constants::kNumActionTypes; ++action_type) {
    return ActionTypeDispatcher::call(action_type, [&](auto AT) {
      int num_actions = mp::ValueAt_v<kNumActionsPerType, AT>;
      for (core::action_t action = 0; action < num_actions; ++action) {
        for (group::element_t sym = 0; sym < Game::SymmetryGroup::kOrder; ++sym) {
          core::action_t transformed_action = action;
          Game::Symmetries::apply(transformed_action, sym);

          PolicyTensorVariant policy_variant(std::in_place_index<AT>);
          auto& policy_tensor = std::get<AT>(policy_variant);
          policy_tensor.setZero();
          policy_tensor(action) = 1;

          PolicyTensorVariant transformed_policy_variant = policy_variant;
          Game::Symmetries::apply(transformed_policy_variant, sym);
          auto& transformed_policy_tensor = std::get<AT>(transformed_policy_variant);

          float sum = eigen_util::sum(transformed_policy_tensor);
          if (sum != 1) {
            printf("Failure in %s() at %s:%d\n", func, __FILE__, __LINE__);
            printf("sym=%d action:%d->%d\n", sym, action, transformed_action);
            printf("Unexpected sum(transformed_policy): %.f\n", sum);
            return false;
          }

          if (transformed_policy_tensor(transformed_action) != 1) {
            printf("Failure in %s() at %s:%d\n", func, __FILE__, __LINE__);
            printf("sym=%d action:%d->%d\n", sym, action, transformed_action);
            printf("Not consistent with transformed policy:\n");
            for (int i = 0; i < Game::Constants::kNumActions; ++i) {
              printf("%s: %.f\n", Game::IO::action_to_str(i).c_str(), transformed_policy_tensor(i));
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
    });
  }

  return true;
}

template <concepts::Game Game>
void Common<Game>::gtest_action_transforms() {
  using Constants = Game::Constants;
  using kNumActionsPerType = Constants::kNumActionsPerType;
  using ActionTypeDispatcher = Game::Types::ActionTypeDispatcher;
  using PolicyTensorVariant = Game::Types::PolicyTensorVariant;

  for (action_type_t action_type = 0; action_type < Constants::kNumActionTypes; ++action_type) {
    ActionTypeDispatcher::call(action_type, [&](auto AT) {
      int num_actions = mp::ValueAt_v<kNumActionsPerType, AT>;
      for (core::action_t action = 0; action < num_actions; ++action) {
        for (group::element_t sym = 0; sym < Game::SymmetryGroup::kOrder; ++sym) {
          core::action_t transformed_action = action;
          Game::Symmetries::apply(transformed_action, sym);

          PolicyTensorVariant policy_variant(std::in_place_index<AT>);
          auto& policy_tensor = std::get<AT>(policy_variant);
          policy_tensor.setZero();
          policy_tensor(action) = 1;

          PolicyTensorVariant transformed_policy_variant = policy_variant;
          Game::Symmetries::apply(transformed_policy_variant, sym);
          auto& transformed_policy_tensor = std::get<AT>(transformed_policy_variant);

          float sum = eigen_util::sum(transformed_policy_tensor);
          EXPECT_FLOAT_EQ(sum, 1);
          EXPECT_FLOAT_EQ(transformed_policy_tensor(transformed_action), 1);

          core::action_t inv_sym = Game::SymmetryGroup::inverse(sym);
          Game::Symmetries::apply(transformed_action, inv_sym);
          EXPECT_EQ(transformed_action, action);

        }
      }
    });
  }
}

}  // namespace tests
}  // namespace core
