#pragma once

#include "core/ActionResponse.hpp"
#include "core/BasicTypes.hpp"
#include "alpha0/concepts/SpecConcept.hpp"
#include "util/EigenUtil.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec EvalSpec>
struct SearchResults;

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
template <alpha0::concepts::Spec EvalSpec>
struct TrainingInfo {
  using Game = EvalSpec::Game;
  using Move = Game::Move;
  using InputFrame = EvalSpec::InputFrame;
  using Types = Game::Types;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using InputEncoder = TensorEncodings::InputEncoder;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using ActionResponse = core::ActionResponse<Game>;

  TrainingInfo() = default;
  TrainingInfo(bool use_for_training, const ActionResponse& response,
               const SearchResults<EvalSpec>* mcts_results, core::seat_index_t seat,
               bool prev_entry_used_for_training);

  void clear() { *this = TrainingInfo(); }

  InputFrame frame;
  PolicyTensor policy_target;
  ActionValueTensor action_values_target;
  Move move;
  core::seat_index_t active_seat;
  bool use_for_training = false;
  bool policy_target_valid = false;
  bool action_values_target_valid = false;

 private:
  static bool validate_and_symmetrize_policy_target(const SearchResults<EvalSpec>* mcts_results,
                                                    PolicyTensor& target);
  static ActionValueTensor apply_mask(const ActionValueTensor& values, const PolicyTensor& mask,
                                      float invalid_value = -1.0f);
};

}  // namespace alpha0

#include "inline/alpha0/TrainingInfo.inl"
