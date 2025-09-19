#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

namespace alpha0 {

/*
 * Whenever use_for_training is true, policy_target_valid and action_values_target_valid should
 * both be true.
 *
 * The reverse, however, is not true: we can have use_for_training false, but have
 * policy_target_valid or action_values_target_valid true. The reason for this is subtle: it's
 * because we have an opponent-reply-policy target. If we sample position 10 of the game, then we
 * want to export the policy target for position 11 (the opponent's reply), even if we don't
 * sample position 11.
 */
template <core::concepts::Game Game>
struct TrainingInfo {
  using State = Game::State;
  using Types = Game::Types;
  using PolicyTensor = Types::PolicyTensor;
  using ActionValueTensor = Types::ActionValueTensor;

  void clear() { *this = TrainingInfo(); }

  State state;
  PolicyTensor policy_target;
  ActionValueTensor action_values_target;
  core::action_t action;
  core::seat_index_t active_seat;
  bool use_for_training = false;
  bool policy_target_valid = false;
  bool action_values_target_valid = false;
};

}  // namespace alpha0
