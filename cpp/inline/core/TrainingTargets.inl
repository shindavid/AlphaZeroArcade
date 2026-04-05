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
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  tensor = view.game_result;
  GameResultEncoding::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool ActionValueTarget<Game>::encode(const GameLogView& view, Tensor& tensor) {
  if (!view.action_values_valid) return false;
  tensor = view.action_values;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool QTarget<Game>::encode(const GameLogView& view, Tensor& tensor) {
  tensor = view.Q;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool QMinTarget<Game>::encode(const GameLogView& view, Tensor& tensor) {
  tensor = view.Q_min;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool QMaxTarget<Game>::encode(const GameLogView& view, Tensor& tensor) {
  tensor = view.Q_max;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool WTarget<Game>::encode(const GameLogView& view, Tensor& tensor) {
  tensor = view.W;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool ActionValueUncertaintyTarget<Game>::encode(const GameLogView& view, Tensor& tensor) {
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
