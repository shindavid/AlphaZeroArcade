#include <mcts/NNEvaluation.hpp>

#include <util/EigenUtil.hpp>
#include <util/Exception.hpp>

namespace mcts {

template <core::concepts::Game Game>
NNEvaluation<Game>::NNEvaluation(const ValueTensor& value, const PolicyTensor& policy,
                                 const ActionValueTensor& action_value,
                                 const ActionMask& valid_actions)
    : dynamic_action_array_(2, valid_actions.count()) {
  int i = 0;
  for (int a : bitset_util::on_indices(valid_actions)) {
    dynamic_action_array_(0, i) = policy(a);
    dynamic_action_array_(1, i) = action_value(a);
    i++;
  }

  dynamic_action_array_.row(0) = eigen_util::softmax(dynamic_action_array_.row(0));
  value_distr_ = eigen_util::softmax(eigen_util::reinterpret_as_array(value));
}

}  // namespace mcts
