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
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using ValueArray = Game::Types::ValueArray;
  using ValueShape = eigen_util::Shape<Game::Constants::kNumPlayers>;
  using ValueTensor = eigen_util::FTensor<ValueShape>;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;

  // 2 rows, one for policy, one for child-value
  using DynamicActionArray = Eigen::Array<float, 2, Eigen::Dynamic>;

  NNEvaluation(const ValueTensor& value, const PolicyTensor& policy,
               const ActionValueTensor& action_values, const ActionMask& valid_actions);
  void load(ValueArray&, LocalPolicyArray&, LocalActionValueArray&);

  using sptr = std::shared_ptr<NNEvaluation>;
  using asptr = util::AtomicSharedPtr<NNEvaluation>;

 protected:
  ValueArray value_distr_;
  DynamicActionArray dynamic_action_array_;
};

}  // namespace mcts

#include <inline/mcts/NNEvaluation.inl>
