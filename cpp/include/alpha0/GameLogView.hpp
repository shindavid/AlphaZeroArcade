#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "util/FiniteGroups.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
struct GameLogCompactRecord;

template <core::concepts::EvalSpec EvalSpec>
struct GameLogView {
  using Game = EvalSpec::Game;
  using InputFrame = EvalSpec::InputFrame;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using GameResultTensor = TensorEncodings::GameResultEncoding::Tensor;

  using GameLogCompactRecord = alpha0::GameLogCompactRecord<EvalSpec>;

  struct Params {
    const GameLogCompactRecord* record = nullptr;
    const GameLogCompactRecord* next_record = nullptr;
    const InputFrame* cur_frame = nullptr;
    const InputFrame* final_frame = nullptr;
    const GameResultTensor* outcome = nullptr;
    group::element_t sym = group::kIdentity;
  };

  GameLogView() = default;
  explicit GameLogView(const Params& params);

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

#include "inline/alpha0/GameLogView.inl"
