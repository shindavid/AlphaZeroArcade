#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
struct GameLogView {
  using Game = EvalSpec::Game;
  using InputFrame = EvalSpec::InputFrame;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using GameResultTensor = TensorEncodings::GameResultEncoding::Tensor;

  InputFrame cur_frame;
  InputFrame final_frame;
  GameResultTensor game_result;
  PolicyTensor policy;
  PolicyTensor next_policy;
  ActionValueTensor action_values;

  core::seat_index_t active_seat;
  bool policy_valid;
  bool next_policy_valid;
  bool action_values_valid;
};

}  // namespace alpha0
