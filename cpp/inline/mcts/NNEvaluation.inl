#include "mcts/NNEvaluation.hpp"

#include "util/Asserts.hpp"
#include "util/BitSet.hpp"
#include "util/EigenUtil.hpp"

namespace mcts {

template <core::concepts::Game Game>
void NNEvaluation<Game>::init(PolicyTensor& policy, ValueTensor& value,
                              ActionValueTensor& action_values, const ActionMask& valid_actions,
                              group::element_t sym, core::seat_index_t active_seat,
                              core::action_mode_t mode) {
  dynamic_array_.resize(2, valid_actions.count());

  // value prediction is from current-player's POV, so rotate it
  value = eigen_util::softmax(value);
  Game::GameResults::right_rotate(value, active_seat);

  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);
  Game::Symmetries::apply(policy, inv_sym, mode);
  Game::Symmetries::apply(action_values, inv_sym, mode);

  int i = 0;
  for (core::action_t a : bitset_util::on_indices(valid_actions)) {
    dynamic_array_(0, i) = policy(a);
    dynamic_array_(1, i) = action_values(a);
    i++;
  }

  dynamic_array_.row(0) = eigen_util::softmax(dynamic_array_.row(0));

  // TODO: this sigmoid() call assumes that the action-values are logits. If/when we devise
  // networks that output non-logit action-values, we should modify this.
  dynamic_array_.row(1) = eigen_util::sigmoid(dynamic_array_.row(1));

  value_ = value;
  eigen_util::debug_assert_is_valid_prob_distr(value_);
  initialized_ = true;
}

template <core::concepts::Game Game>
void NNEvaluation<Game>::uniform_init(const ActionMask& valid_actions) {
  dynamic_array_.resize(2, valid_actions.count());

  float policy_entry = 1.0 / valid_actions.count();
  float value_entry = 1.0 / value_.size();
  float action_value_entry = 1.0 / Game::Constants::kNumPlayers;

  value_.setConstant(value_entry);
  dynamic_array_.row(0).setConstant(policy_entry);
  dynamic_array_.row(1).setConstant(action_value_entry);
  initialized_ = true;
}

template <core::concepts::Game Game>
bool NNEvaluation<Game>::decrement_ref_count() {
  // NOTE: during normal program execution, this is performed in a thread-safe manner. On the
  // other hand, when the program is shutting down, it is not. Thankfully, we don't require thread
  // safety during that phase of the program. If for some reason that changes, we will need to
  // use std::atomic
  ref_count_--;
  return ref_count_ == 0;
}

template <core::concepts::Game Game>
void NNEvaluation<Game>::clear() {
  aux_ = nullptr;
  eval_sequence_id_ = 0;
  ref_count_ = 0;
  initialized_ = false;
}

template <core::concepts::Game Game>
void NNEvaluation<Game>::load(ValueTensor& value, LocalPolicyArray& policy,
                              LocalActionValueArray& action_value) {
  RELEASE_ASSERT(initialized_, "NNEvaluation not initialized");
  value = value_;
  policy = dynamic_array_.row(0);
  action_value = dynamic_array_.row(1);
  eigen_util::debug_assert_is_valid_prob_distr(policy);
}

}  // namespace mcts
