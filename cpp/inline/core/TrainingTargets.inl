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
bool FutureMCTSValueTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  if (!view.future_mcts_value_valid) return false;
  tensor = view.future_mcts_value;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool ActionValueUncertaintyTarget<TensorEncodings>::encode(const GameLogView& view,
                                                           Tensor& tensor) {
  if (!view.action_values_valid) return false;
  tensor = view.AU + 1e-8f;  // avoid vanishing gradient pathology, matching KataGo
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
