#pragma once

#include <memory>

#include <core/concepts/Game.hpp>
#include <util/AtomicSharedPtr.hpp>
#include <util/EigenUtil.hpp>

namespace mcts {

template <core::concepts::Game Game>
class NNEvaluation {
 public:
  using ActionMask = Game::Types::ActionMask;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;
  using ValueShape = eigen_util::Shape<Game::Constants::kNumPlayers>;
  using ValueTensor = eigen_util::FTensor<ValueShape>;
  using CompactLocalPolicyArray = Game::Types::CompactLocalPolicyArray;

  NNEvaluation(const ValueTensor& value, const PolicyTensor& policy,
               const ActionMask& valid_actions);
  const ValueArray& value_distr() const { return value_distr_; }
  CompactLocalPolicyArray compact_local_policy_distr() const { return compact_local_policy_distr_; }

  using sptr = std::shared_ptr<NNEvaluation>;
  using asptr = util::AtomicSharedPtr<NNEvaluation>;

 protected:
  ValueArray value_distr_;
  CompactLocalPolicyArray compact_local_policy_distr_;
};

}  // namespace mcts

#include <inline/mcts/NNEvaluation.inl>
