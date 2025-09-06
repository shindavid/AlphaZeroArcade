#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "util/FiniteGroups.hpp"

#include <Eigen/Core>

namespace nnet {

// TODO: We could expand the template params of this class to include Traits::EvalServiceBase.
// That would allow us to replace the void* aux blob with something more specific.
template <core::concepts::EvalSpec EvalSpec>
class NNEvaluation {
 public:
  using Game = EvalSpec::Game;
  using InputTensorizor = Game::InputTensorizor;
  using TrainingTargets = EvalSpec::TrainingTargets;

  using ActionValueTarget = TrainingTargets::ActionValueTarget;
  using ActionMask = Game::Types::ActionMask;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueTensor = Game::Types::ValueTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;

  // 2 rows, one for policy, one for action-value
  using DynamicArray = Eigen::Array<float, 2, Eigen::Dynamic>;

  /*
   * Warning: the passed-in tensors are modified in-place!
   *
   * The tensors passed-in are raw tensors from the neural network. To convert them into usable
   * tensors, the constructor performs the following:
   *
   * 1. Transform from logit space to probability space via softmax()
   * 2. Undo applied symmetry (sym) where appropriate
   * 3. Rotate value to align with the active seat
   *
   * These tensors are then stored as data members.
   */
  void init(PolicyTensor&, ValueTensor&, ActionValueTensor&, const ActionMask& valid_actions,
            group::element_t sym, core::seat_index_t active_seat, core::action_mode_t mode);

  void uniform_init(const ActionMask&);  // Used by UniformNNEvaluationService

  bool decrement_ref_count();  // returns true iff ref_count_ == 0
  void increment_ref_count() { ref_count_++; }
  int ref_count() const { return ref_count_; }
  bool pending() const { return !initialized_; }
  void clear();

  void set_aux(void* aux) { aux_ = aux; }
  template <typename T>
  T* get_aux() {
    return static_cast<T*>(aux_);
  }

  void set_eval_sequence_id(core::nn_evaluation_sequence_id_t id) { eval_sequence_id_ = id; }
  core::nn_evaluation_sequence_id_t eval_sequence_id() const { return eval_sequence_id_; }

  void load(ValueTensor&, LocalPolicyArray&, LocalActionValueArray&);

 protected:
  ValueTensor value_;
  DynamicArray dynamic_array_;
  void* aux_ = nullptr;  // set to a NNEvaluationService-specific object
  core::nn_evaluation_sequence_id_t eval_sequence_id_ = 0;
  int ref_count_ = 0;  // access only permitted under NNEvaluationService cache_mutex_!
  bool initialized_ = false;
};

}  // namespace nnet

#include "inline/nnet/NNEvaluation.inl"
