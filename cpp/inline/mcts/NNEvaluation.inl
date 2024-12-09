#include <mcts/NNEvaluation.hpp>

#include <util/BitSet.hpp>
#include <util/EigenUtil.hpp>
#include <util/Exception.hpp>

namespace mcts {

template <core::concepts::Game Game>
NNEvaluation<Game>::NNEvaluation(const ValueTensor& raw_value, const PolicyTensor& raw_policy,
                                 const ActionValueTensor& raw_action_values,
                                 const ActionMask& valid_actions, group::element_t sym,
                                 core::seat_index_t cp)
    : dynamic_array_(2, get_num_valid_actions(valid_actions)) {
  ValueTensor value = raw_value;
  PolicyTensor policy = raw_policy;
  ActionValueTensor action_values = raw_action_values;

  // value prediction is from current-player's POV, so rotate it
  value = eigen_util::softmax(value);
  Game::GameResults::right_rotate(value, cp);

  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);
  Game::Symmetries::apply(policy, inv_sym);
  Game::Symmetries::apply(action_values, inv_sym);

  ActionTypeDispatcher::call(valid_actions.index(), [&](auto action_type) {
    int i = 0;
    for (core::action_t a : bitset_util::on_indices(std::get<action_type>(valid_actions))) {
      dynamic_array_(0, i) = policy(a);
      dynamic_array_(1, i) = action_values(a);
      i++;
    }
  });

  dynamic_array_.row(0) = eigen_util::softmax(dynamic_array_.row(0));

  // TODO: this sigmoid() call assumes that the action-values are logits. If/when we devise
  // networks that output non-logit action-values, we should modify this.
  dynamic_array_.row(1) = eigen_util::sigmoid(dynamic_array_.row(1));

  value_ = value;
  eigen_util::debug_assert_is_valid_prob_distr(value_);
}

template <core::concepts::Game Game>
NNEvaluation<Game>::NNEvaluation(const ActionMask& valid_actions)
    : dynamic_array_(2, get_num_valid_actions(valid_actions)) {
  int n_cols = dynamic_array_.cols();
  float policy_entry = 1.0 / n_cols;
  float value_entry = 1.0 / value_.size();
  float action_value_entry = 1.0 / Game::Constants::kNumPlayers;

  value_.setConstant(value_entry);
  dynamic_array_.row(0).setConstant(policy_entry);
  dynamic_array_.row(1).setConstant(action_value_entry);
}

template <core::concepts::Game Game>
void NNEvaluation<Game>::load(ValueTensor& value, LocalPolicyArray& policy,
                              LocalActionValueArray& action_value) {
  value = value_;
  policy = dynamic_array_.row(0);
  action_value = dynamic_array_.row(1);
  eigen_util::debug_assert_is_valid_prob_distr(policy);
}

template <core::concepts::Game Game>
typename NNEvaluation<Game>::sptr NNEvaluation<Game>::create_uniform(
    const ActionMask& valid_actions) {
  return std::make_shared<NNEvaluation>(valid_actions);
}

template <core::concepts::Game Game>
int NNEvaluation<Game>::get_num_valid_actions(const ActionMask& valid_actions) {
  return ActionTypeDispatcher::call(valid_actions.index(), [&](auto action_type) {
    return std::get<action_type>(valid_actions).count();
  });
}

}  // namespace mcts
