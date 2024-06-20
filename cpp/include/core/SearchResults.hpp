#pragma once

namespace core {

template <typename GameTypes>
struct SearchResults {
  using ActionMask = typename GameTypes::ActionMask;
  using PolicyTensor = typename GameTypes::PolicyTensor;
  using ValueArray = typename GameTypes::ValueArray;

  ActionMask valid_actions;
  PolicyTensor counts;
  PolicyTensor policy_target;
  PolicyTensor policy_prior;
  ValueArray win_rates;
  ValueArray value_prior;
  bool provably_lost = false;
};

}  // namespace core
