#pragma once

#include "core/BasicTypes.hpp"
#include "core/InputTensorizor.hpp"
#include "core/TensorTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/FiniteGroups.hpp"

#include <Eigen/Core>

namespace search {

// TODO: We could expand the template params of this class to include Traits::EvalServiceBase.
// That would allow us to replace the void* aux blob with something more specific.
template <search::concepts::Traits Traits>
class NNEvaluation {
 public:
  using Game = Traits::Game;
  using EvalSpec = Traits::EvalSpec;
  using TensorTypes = core::TensorTypes<EvalSpec>;
  using InputTensorizor = core::InputTensorizor<Game>;
  using NetworkHeads = Traits::EvalSpec::NetworkHeads::List;

  static constexpr int kNumOutputs = TensorTypes::kNumOutputs;

  using ActionMask = Game::Types::ActionMask;
  using OutputTensorTuple = TensorTypes::OutputTensorTuple;
  using LocalPolicyTensor = Game::Types::LocalPolicyTensor;

  ~NNEvaluation() { clear(); }

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
  void init(OutputTensorTuple& outputs, const ActionMask& valid_actions, group::element_t sym,
            core::seat_index_t active_seat, core::action_mode_t mode);

  void uniform_init(const ActionMask&);  // Used by UniformNNEvaluationService

  bool decrement_ref_count();  // returns true iff ref_count_ == 0
  void increment_ref_count() { ref_count_++; }
  int ref_count() const { return ref_count_; }
  bool pending() const { return !data_; }
  void clear();

  void set_aux(void* aux) { aux_ = aux; }
  template <typename T>
  T* get_aux() {
    return static_cast<T*>(aux_);
  }

  void set_eval_sequence_id(core::nn_evaluation_sequence_id_t id) { eval_sequence_id_ = id; }
  core::nn_evaluation_sequence_id_t eval_sequence_id() const { return eval_sequence_id_; }

  const float* data(int index) const { return data_helper(data_, index); }
  float* data(int index) { return data_helper(data_, index); }

 protected:
  const float* data_helper(float* d, int i) const { return d + (i == 0 ? 0 : offsets_[i - 1]); }
  float* data_helper(float* d, int i) { return d + (i == 0 ? 0 : offsets_[i - 1]); }

  float* init_data_and_offsets(const ActionMask& valid_actions);

  float* data_ = nullptr;
  void* aux_ = nullptr;  // set to a NNEvaluationService-specific object
  core::nn_evaluation_sequence_id_t eval_sequence_id_ = 0;
  int offsets_[kNumOutputs - 1];  // leave off trivial 0
  int ref_count_ = 0;             // access only permitted under NNEvaluationService cache_mutex_!
};

}  // namespace search

#include "inline/search/NNEvaluation.inl"
