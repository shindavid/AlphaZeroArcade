#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationRequest.hpp>
#include <mcts/NNEvaluationServiceBase.hpp>

template <core::concepts::Game Game>
class UniformNNEvaluationService : public mcts::NNEvaluationServiceBase<Game> {
 public:
  using NNEvaluation = mcts::NNEvaluation<Game>;
  using NNEvaluationRequest = mcts::NNEvaluationRequest<Game>;
  using ValueTensor = NNEvaluation::ValueTensor;
  using PolicyTensor = NNEvaluation::PolicyTensor;
  using ActionValueTensor = NNEvaluation::ActionValueTensor;
  using ActionMask = NNEvaluation::ActionMask;

  void evaluate(const NNEvaluationRequest& request) override {
    ValueTensor value;
    PolicyTensor policy;
    ActionValueTensor action_values;
    group::element_t sym = group::kIdentity;

    for (typename NNEvaluationRequest::Item& item : request.items()) {
      ActionMask valid_actions = item.node()->stable_data().valid_action_mask;
      core::seat_index_t cp = item.node()->stable_data().current_player;

      int n = valid_actions.count();
      float p = 1.0 / n;

      policy.setZero();
      for (int i = 0; i < policy.size(); i++) {
        if (valid_actions[i]) {
          policy[i] = p;
        }
      }
      value.setZero();
      action_values.setZero();

      item.set_eval(
          std::make_shared<NNEvaluation>(value, policy, action_values, valid_actions, sym, cp));
    }
  }
};