#include "core/TrainingTargets.hpp"

namespace core {

template <core::concepts::Game Game>
template <typename GameLogView>
bool PolicyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.policy_valid) return false;
  tensor = view.policy;
  return true;
}

template <core::concepts::Game Game>
template <typename Derived>
void PolicyTarget<Game>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::softmax_in_place(dst);
}

template <core::concepts::Game Game>
template <typename Derived>
void PolicyTarget<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  dst.setConstant(1.0 / eigen_util::size(dst));
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool ValueTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor = view.game_result;
  Game::GameResults::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename Derived>
void ValueTarget<Game>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::softmax_in_place(dst);
}

template <core::concepts::Game Game>
template <typename Derived>
void ValueTarget<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  dst.setConstant(1.0 / eigen_util::size(dst));
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool ActionValueTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (view.action_values_valid) return false;
  tensor = view.action_values;
  return true;
}

template <core::concepts::Game Game>
template <typename Derived>
void ActionValueTarget<Game>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::sigmoid_in_place(dst);
}

template <core::concepts::Game Game>
template <typename Derived>
void ActionValueTarget<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  dst.setConstant(1.0 / Game::Constants::kNumPlayers);
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool ValueUncertaintyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  float x = (view.Q_prior - view.Q_posterior);
  tensor(0) = x * x;
  return true;
}

template <core::concepts::Game Game>
template <typename Derived>
void ValueUncertaintyTarget<Game>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::sigmoid_in_place(dst);
}

template <core::concepts::Game Game>
template <typename Derived>
void ValueUncertaintyTarget<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  constexpr float p = 1.0f / Game::Constants::kNumPlayers;
  dst.setConstant(p * (1 - p));
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool ActionValueUncertaintyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (view.action_value_uncertainties_valid) return false;
  tensor = view.action_value_uncertainties;
  return true;
}

template <core::concepts::Game Game>
template <typename Derived>
void ActionValueUncertaintyTarget<Game>::transform(Eigen::TensorBase<Derived>& dst) {
  eigen_util::sigmoid_in_place(dst);
}

template <core::concepts::Game Game>
template <typename Derived>
void ActionValueUncertaintyTarget<Game>::uniform_init(Eigen::TensorBase<Derived>& dst) {
  constexpr float p = 1.0f / Game::Constants::kNumPlayers;
  dst.setConstant(p * (1 - p));
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool OppPolicyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.next_policy_valid) return false;
  tensor = view.next_policy;
  return true;
}

}  // namespace core
