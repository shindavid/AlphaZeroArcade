#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/FiniteGroups.hpp"
#include "util/MetaProgramming.hpp"

#include <Eigen/Core>

#include <tuple>

namespace search {

namespace detail {

template <typename Head>
struct ExtractTensor {
  using type = Head::Tensor;
};

}  // namespace detail

template <core::concepts::Game Game_, typename InputFrame_, typename NetworkHeads_>
class NNEvaluation {
 public:
  using Game = Game_;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;
  using InputFrame = InputFrame_;
  using NetworkHeads = NetworkHeads_;

  using OutputTensors = mp::Apply_t<NetworkHeads, detail::ExtractTensor>;
  using OutputTensorTuple = mp::Rebind_t<OutputTensors, std::tuple>;

  static constexpr int kNumOutputs = mp::Length_v<OutputTensors>;

  struct InitParams {
    OutputTensorTuple& outputs;
    const MoveSet& valid_moves;
    const InputFrame& frame;
    group::element_t sym;
    core::seat_index_t active_seat;
  };

  ~NNEvaluation() { clear(); }

  // Warning: the passed-in tensors are modified in-place before being packed into this object's
  // aligned storage.
  void init(const InitParams& params);

  void uniform_init(int num_valid_moves);  // Used by UniformNNEvaluationService

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

  float* init_data_and_offsets(int num_valid_moves);

  float* data_ = nullptr;
  void* aux_ = nullptr;  // set to a NNEvaluationService-specific object
  core::nn_evaluation_sequence_id_t eval_sequence_id_ = 0;
  int offsets_[kNumOutputs - 1];  // leave off trivial 0
  int ref_count_ = 0;             // access only permitted under NNEvaluationService cache_mutex_!
};

}  // namespace search

#include "inline/search/NNEvaluation.inl"
