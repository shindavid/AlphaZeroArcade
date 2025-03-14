#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <util/AtomicSharedPtr.hpp>
#include <util/FiniteGroups.hpp>

#include <memory>

namespace mcts {

template <core::concepts::Game Game>
class NNEvaluation {
 public:
  using ActionValueTarget = Game::TrainingTargets::ActionValueTarget;
  using ActionMask = Game::Types::ActionMask;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueTensor = Game::Types::ValueTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;

  // 2 rows, one for policy, one for action-value
  using DynamicArray = Eigen::Array<float, 2, Eigen::Dynamic>;

  /*
   * The tensors passed-in are raw tensors from the neural network. To convert them into usable
   * tensors, the constructor performs the following:
   *
   * 1. Transform from logit space to probability space via softmax()
   * 2. Undo applied symmetry (sym) where appropriate
   * 3. Rotate value to align with the active seat
   *
   * These tensors are then stored as data members.
   */
  NNEvaluation(const ValueTensor& raw_value, const PolicyTensor& raw_policy,
               const ActionValueTensor& raw_action_values,
               const ActionMask& valid_actions, group::element_t sym,
               core::seat_index_t active_seat, core::action_mode_t mode);

  // This constructor is used by create_uniform(). We would declare this as private if we could,
  // but can't because std::make_shared<> needs to call it.
  NNEvaluation(const ActionMask&);  // used by create_uniform()

  void load(ValueTensor&, LocalPolicyArray&, LocalActionValueArray&);

  using sptr = std::shared_ptr<NNEvaluation>;
  using asptr = util::AtomicSharedPtr<NNEvaluation>;

  /*
   * Create a uniform NNEvaluation object. Used by UniformNNEvaluationService.
   */
  static sptr create_uniform(const ActionMask& valid_actions);

 protected:
  ValueTensor value_;
  DynamicArray dynamic_array_;
};

}  // namespace mcts

#include <inline/mcts/NNEvaluation.inl>
