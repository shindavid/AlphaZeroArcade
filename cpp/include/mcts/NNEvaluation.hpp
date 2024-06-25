#pragma once

#include <memory>

#include <core/concepts/Game.hpp>
#include <util/AtomicSharedPtr.hpp>

namespace mcts {

template <core::concepts::Game Game>
class NNEvaluation {
 public:
  using ActionMask = Game::Types::ActionMask;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;
  using LocalPolicyArray = eigen_util::DArray<Game::Constants::kMaxBranchingFactor>;
  using ValueShape = eigen_util::Shape<Game::Constants::kNumPlayers>;
  using ValueTensor = eigen_util::FTensor<ValueShape>;

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

#include <inline/mcts/NNEvaluation.inl>
