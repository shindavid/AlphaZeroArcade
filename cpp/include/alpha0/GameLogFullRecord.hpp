#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
struct GameLogFullRecord {
  using InputEncoder = EvalSpec::TensorEncodings::InputEncoder;
  using InputFrame = EvalSpec::InputFrame;
  using PolicyTensor = EvalSpec::TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = EvalSpec::Game::Types::ActionValueTensor;
  using Move = EvalSpec::Game::Move;

  InputFrame frame;
  PolicyTensor policy_target;       // only valid if policy_target_valid
  ActionValueTensor action_values;  // only valid if action_values_valid
  Move move;
  core::game_phase_t game_phase;
  core::seat_index_t active_seat;
  bool use_for_training;
  bool policy_target_valid;
  bool action_values_valid;
};

}  // namespace alpha0
