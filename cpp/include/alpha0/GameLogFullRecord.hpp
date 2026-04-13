#pragma once

#include "alpha0/GameLogCompactRecord.hpp"
#include "core/BasicTypes.hpp"
#include "alpha0/concepts/EvalSpecConcept.hpp"
#include "search/GameLogBase.hpp"

#include <vector>

namespace alpha0 {

template <alpha0::concepts::EvalSpec EvalSpec>
struct TrainingInfo;

template <alpha0::concepts::EvalSpec EvalSpec>
struct GameLogFullRecord {
  using InputFrame = EvalSpec::InputFrame;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using InputEncoder = TensorEncodings::InputEncoder;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using Move = EvalSpec::Game::Move;

  using PolicyShape = TensorEncodings::PolicyEncoding::Shape;
  using ActionValueShape = TensorEncodings::ActionValueEncoding::Shape;
  using PolicyTensorData = search::TensorData<PolicyShape>;
  using ActionValueTensorData = search::TensorData<ActionValueShape>;
  using GameLogCompactRecord = alpha0::GameLogCompactRecord<EvalSpec>;

  GameLogFullRecord() = default;
  explicit GameLogFullRecord(const TrainingInfo<EvalSpec>&);
  void serialize(std::vector<char>& buf) const;

  InputFrame frame;
  PolicyTensor policy_target;       // only valid if policy_target_valid
  ActionValueTensor action_values;  // only valid if action_values_valid
  Move move;
  core::seat_index_t active_seat;
  bool use_for_training;
  bool policy_target_valid;
  bool action_values_valid;
};

}  // namespace alpha0

#include "inline/alpha0/GameLogFullRecord.inl"
