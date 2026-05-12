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

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool SsStarTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  if (!view.backup_sample.valid) return false;
  tensor = view.backup_sample.Ss_star;
  return true;
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool WsStarTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  if (!view.backup_sample.valid) return false;
  tensor(0) = view.backup_sample.Ws_star;
  return true;
}

template <core::concepts::TensorEncodings TensorEncodings>
template <typename GameLogView>
bool ChildStatsTarget<TensorEncodings>::encode(const GameLogView& view, Tensor& tensor) {
  if (!view.backup_sample.valid) return false;
  // FTensor is RowMajor with the channel dim appended last, so the flat layout is
  // [a0_c0, a0_c1, ..., a0_c5, a1_c0, ...]. We can pack channel-by-channel via flat indexing.
  using PolicyTensor = std::remove_cvref_t<decltype(view.backup_sample.N)>;
  constexpr int A = PolicyTensor::Dimensions::total_size;
  float* dst = tensor.data();
  const float* N = view.backup_sample.N.data();
  const float* Qs = view.backup_sample.Qs.data();
  const float* Ws = view.backup_sample.Ws.data();
  const float* P = view.backup_sample.P.data();
  const float* AVs = view.backup_sample.AVs.data();
  const float* AUs = view.backup_sample.AUs.data();
  for (int a = 0; a < A; ++a) {
    dst[a * kNumChildStats + 0] = Qs[a];
    dst[a * kNumChildStats + 1] = Ws[a];
    dst[a * kNumChildStats + 2] = N[a];
    dst[a * kNumChildStats + 3] = P[a];
    dst[a * kNumChildStats + 4] = AVs[a];
    dst[a * kNumChildStats + 5] = AUs[a];
  }
  return true;
}

}  // namespace core
