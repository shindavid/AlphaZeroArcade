#include "core/TrainingTargets.hpp"

namespace core {

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool PolicyTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  if (!view.policy_valid) return false;
  tensor = view.policy;
  return true;
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool ValueTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  tensor = view.game_result;
  GameResultEncoding::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool ActionValueTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  if (!view.action_values_valid) return false;
  tensor = view.action_values;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool QTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  tensor = view.Q;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool QMinTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  tensor = view.Q_min;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool QMaxTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  tensor = view.Q_max;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool WTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  tensor = view.W;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool ActionValueUncertaintyTarget<TensorEncodings>::encode(const GameLogView& view,
                                                           Tensor& tensor) {
  if (!view.action_values_valid) return false;
  tensor = view.AU;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool OppPolicyTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  if (!view.next_policy_valid) return false;
  tensor = view.next_policy;
  return true;
}

}  // namespace core
