#pragma once

#include "alpha0/concepts/SpecConcept.hpp"
#include "core/BasicTypes.hpp"
#include "util/FiniteGroups.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
struct GameLogCompactRecord;

template <alpha0::concepts::Spec Spec>
struct GameLogView {
  using Game = Spec::Game;
  using InputFrame = Spec::InputFrame;
  using TensorEncodings = Spec::TensorEncodings;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using GameResultTensor = TensorEncodings::GameResultEncoding::Tensor;

  using GameLogCompactRecord = alpha0::GameLogCompactRecord<Spec>;

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
