#pragma once

#include "beta0/concepts/SpecConcept.hpp"
#include "core/ActionResponse.hpp"
#include "core/BasicTypes.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
struct SearchResults;

/*
 * TrainingInfo for BetaZero.
 *
 * Compared to alpha0::TrainingInfo, this adds:
 *   - Q_root: the search Q at the time add() is called (stored for retroactive W computation)
 *   - action_values_uncertainty_target: per-action AU target
 *
 * W_target is retroactively computed in GameWriteLog::add_terminal() via a lambda-discounted
 * sum of future Q_root values (KataGo LoTV formulation):
 *
 *   W_target[t] = (Q_root[t] - S[t])^2
 *   S[t] = (1-lambda) * Q_root[t+1] + lambda * S[t+1]   (backward pass)
 *   lambda = 5/6
 *
 * Whenever use_for_training is true, policy_target_valid and action_values_target_valid should
 * both be true.
 */
template <beta0::concepts::Spec Spec>
struct TrainingInfo {
  using Game = Spec::Game;
  using Move = Game::Move;
  using InputFrame = Spec::InputFrame;
  using Types = Game::Types;
  using TensorEncodings = Spec::TensorEncodings;
  using InputEncoder = TensorEncodings::InputEncoder;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using ValueArray = Game::Types::ValueArray;
  using ActionResponse = core::ActionResponse<Game>;

  TrainingInfo() = default;
  TrainingInfo(bool use_for_training, const ActionResponse& response,
               const SearchResults<Spec>* mcts_results, core::seat_index_t seat,
               bool prev_entry_used_for_training);

  void clear() { *this = TrainingInfo(); }

  InputFrame frame;
  PolicyTensor policy_target;
  ActionValueTensor action_values_target;
  ActionValueTensor action_values_uncertainty_target;  // AU target
  ValueArray Q_root;                                   // search Q at this step (for W computation)
  Move move;
  core::seat_index_t active_seat;
  bool use_for_training = false;
  bool policy_target_valid = false;
  bool action_values_target_valid = false;

 private:
  static bool validate_and_symmetrize_policy_target(const SearchResults<Spec>* mcts_results,
                                                    PolicyTensor& target);
  static ActionValueTensor apply_mask(const ActionValueTensor& values, const PolicyTensor& mask,
                                      float invalid_value = -1.0f);
};

}  // namespace beta0

#include "inline/beta0/TrainingInfo.inl"
