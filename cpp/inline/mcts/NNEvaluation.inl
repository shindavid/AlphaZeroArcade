#include <mcts/NNEvaluation.hpp>

#include <util/BitSet.hpp>
#include <util/EigenUtil.hpp>
#include <util/Exception.hpp>

namespace mcts {

template <core::concepts::Game Game>
NNEvaluation<Game>::NNEvaluation(const ValueTensor& raw_value, const PolicyTensor& raw_policy,
                                 const FullActionValueTensor& raw_full_action_values,
                                 const ActionMask& valid_actions, group::element_t sym,
                                 core::seat_index_t cp)
    : dynamic_array_(2, valid_actions.count()) {
  ValueTensor value = raw_value;
  PolicyTensor policy = raw_policy;
  ActionValueTensor action_values;

  // Leave out the last entry of raw_full_action_values, which is the invalid action
  int j = 0;
  eigen_util::apply_per_slice<0>(action_values, [&](auto& slice) {
    slice = raw_full_action_values.chip(j, 0);
    j++;
  });

  // value prediction is from current-player's POV, so rotate it
  value = eigen_util::softmax(value);
  Game::GameResults::right_rotate(value, cp);

  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);
  Game::Symmetries::apply(policy, inv_sym);
  eigen_util::apply_per_slice<0>(action_values, [&](auto& slice) {
    using SliceT = std::decay_t<decltype(slice)>;
    static_assert(std::is_same_v<SliceT, PolicyTensor>);
    Game::Symmetries::apply(slice, inv_sym);
  });

  int i = 0;
  for (core::action_t a : bitset_util::on_indices(valid_actions)) {
    dynamic_array_(0, i) = policy(a);
    ValueTensor sub_values = action_values.chip(a, 1);
    sub_values = eigen_util::softmax(sub_values);

    // action-value prediction is already from current-player's POV, so no need to rotate
    dynamic_array_(1, i) = Game::GameResults::to_value_array(sub_values)(0);
    i++;
  }

  dynamic_array_.row(0) = eigen_util::softmax(dynamic_array_.row(0));
  value_ = value;
}

template <core::concepts::Game Game>
void NNEvaluation<Game>::load(ValueTensor& value, LocalPolicyArray& policy,
                              LocalActionValueArray& action_value) {
  value = value_;
  policy = dynamic_array_.row(0);
  action_value = dynamic_array_.row(1);
}

}  // namespace mcts
