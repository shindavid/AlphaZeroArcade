#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace alpha0 {

template <core::concepts::Game Game>
struct GameLogView {
  using State = Game::State;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using GameResultTensor = Game::Types::GameResultTensor;

  State cur_pos;
  State final_pos;
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
