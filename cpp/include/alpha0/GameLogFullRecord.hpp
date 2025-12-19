#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace alpha0 {

template <core::concepts::Game Game>
struct GameLogFullRecord {
  using State = Game::State;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;

  State position;
  PolicyTensor policy_target;       // only valid if policy_target_valid
  ActionValueTensor action_values;  // only valid if action_values_valid
  core::action_t action;
  core::seat_index_t active_seat;
  bool use_for_training;
  bool policy_target_valid;
  bool action_values_valid;
};

}  // namespace alpha0
