#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace alpha0 {

template <core::concepts::Game Game>
struct GameLogView {
  using State = Game::State;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using ValueTensor = Game::Types::ValueTensor;

  State cur_pos;
  State final_pos;
  ValueTensor game_result;
  PolicyTensor policy;
  PolicyTensor next_policy;
  ActionValueTensor action_values;
  core::seat_index_t active_seat;
};

}  // namespace alpha0
