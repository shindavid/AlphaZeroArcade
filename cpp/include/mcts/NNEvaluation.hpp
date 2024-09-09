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

  // 2 rows, one for policy, one for child-value
  using DynamicActionArray = Eigen::Array<float, 2, Eigen::Dynamic>;

  NNEvaluation(const ValueTensor& value, const PolicyTensor& policy,
               const ActionValueTensor& action_values, const ActionMask& valid_actions);
  const ValueArray& value_distr() const { return value_distr_; }
  const DynamicActionArray& dynamic_action_array() const { return dynamic_action_array_; }

  using sptr = std::shared_ptr<NNEvaluation>;
  using asptr = util::AtomicSharedPtr<NNEvaluation>;

 protected:
  ValueArray value_distr_;
  DynamicActionArray dynamic_action_array_;
};

}  // namespace mcts

#include <inline/mcts/NNEvaluation.inl>
