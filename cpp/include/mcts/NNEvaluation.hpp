#pragma once

#include <memory>

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <util/AtomicSharedPtr.hpp>

namespace mcts {

template <core::GameStateConcept GameState>
class NNEvaluation {
 public:
  using GameStateTypes = core::GameStateTypes<GameState>;

  using ActionMask = typename GameStateTypes::ActionMask;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueArray = typename GameStateTypes::ValueArray;
  using ValueTensor = typename GameStateTypes::ValueTensor;

  NNEvaluation(const ValueTensor& value, const PolicyTensor& policy,
               const ActionMask& valid_actions);
  const ValueArray& value_prob_distr() const { return value_prob_distr_; }
  const LocalPolicyArray& local_policy_logit_distr() const { return local_policy_logit_distr_; }

  using sptr = std::shared_ptr<NNEvaluation>;
  using asptr = util::AtomicSharedPtr<NNEvaluation>;

 protected:
  ValueArray value_prob_distr_;
  LocalPolicyArray local_policy_logit_distr_;
};

}  // namespace mcts

#include <mcts/inl/NNEvaluation.inl>
