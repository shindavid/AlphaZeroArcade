#include "beta0/BackupNNEvaluator.hpp"

#include "core/WinLossDrawEncoding.hpp"  // IWYU pragma: keep (for to_value_array)
#include "util/Asserts.hpp"
#include "util/EigenUtil.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace beta0 {

namespace detail {

// Copies an nnue/<key>.<which> tensor's flat float data into `dst.data()`. Asserts the source
// vector has exactly the expected number of floats. Used by reload_weights.
template <typename Dst>
inline void load_named_tensor(const core::ModelBundle& model, const std::string& key, Dst& dst,
                              std::ptrdiff_t expected_count) {
  auto it = model.nnue_weights.find(key);
  RELEASE_ASSERT(it != model.nnue_weights.end(),
                 "BackupNNEvaluator::reload_weights: missing nnue/{} in ModelBundle", key);
  const auto& vec = it->second;
  RELEASE_ASSERT(static_cast<std::ptrdiff_t>(vec.size()) == expected_count,
                 "BackupNNEvaluator::reload_weights: nnue/{} has {} floats, expected {}", key,
                 vec.size(), expected_count);
  std::copy_n(vec.data(), expected_count, dst.data());
}

// S/W-skip clamp. MUST match py/shared/backup_net.py (SSTAR_CLAMP_EPS) byte-for-byte. The
// equivalence unit test (cpp/src/integration_tests/main/BackupNNEquivalenceTests.cpp)
// verifies this by running both sides on a Python-exported ONNX file.
constexpr float kSstarClampEps = 1e-4f;

}  // namespace detail

template <beta0::concepts::Spec Spec>
void BackupNNEvaluator<Spec>::reload_weights(const core::ModelBundle& model) {
  detail::load_named_tensor(model, "child_embed.weight", W_child_embed_, W_child_embed_.size());
  detail::load_named_tensor(model, "child_embed.bias", b_child_embed_, b_child_embed_.size());
  detail::load_named_tensor(model, "layer1.weight", W_l1_, W_l1_.size());
  detail::load_named_tensor(model, "layer1.bias", b_l1_, b_l1_.size());
  detail::load_named_tensor(model, "layer2.weight", W_l2_, W_l2_.size());
  detail::load_named_tensor(model, "layer2.bias", b_l2_, b_l2_.size());
  detail::load_named_tensor(model, "out.weight", W_out_, W_out_.size());
  detail::load_named_tensor(model, "out.bias", b_out_, b_out_.size());
  ready_ = true;
  ++weight_gen_;
}

template <beta0::concepts::Spec Spec>
typename BackupNNEvaluator<Spec>::EmbedArray BackupNNEvaluator<Spec>::compute_child_embedding(
  const ChildStatArray& cs, const ActionLatentArray& za) const {
  // Build x = [cs ; za] as a column vector of length kPerChildInDim.
  Eigen::Matrix<float, kPerChildInDim, 1> x;
  x.template head<kChildStatDim>() = cs.matrix();
  x.template tail<kActionLatentDim>() = za.matrix();

  // pre = W_child_embed_ @ x + b_child_embed_
  Eigen::Array<float, kEmbedDim, 1> pre = (W_child_embed_ * x).array() + b_child_embed_;

  // ReLU + (P > 0) mask
  EmbedArray e = pre.max(0.0f);
  if (cs(kPolicyPriorIndex) <= 0.0f) {
    e.setZero();
  }
  return e;
}

template <beta0::concepts::Spec Spec>
typename BackupNNEvaluator<Spec>::ActiveSeatResult BackupNNEvaluator<Spec>::apply(
  const AccumulatorArray& acc, const StaticLatentArray& z_s, const Tensor& S_baseline,
  float Ws_baseline) const {
  // h0 = [acc ; z_s ; Ss* ; Ws*]  (column vector, length kBackupLayer1InDim)
  Eigen::Matrix<float, kBackupLayer1InDim, 1> h0;
  h0.template head<kEmbedDim>() = acc.matrix();
  h0.template segment<kStaticLatentDim>(kEmbedDim) = z_s.matrix();
  for (int i = 0; i < kValueDim; ++i) {
    h0(kEmbedDim + kStaticLatentDim + i) = S_baseline(i);
  }
  h0(kEmbedDim + kStaticLatentDim + kValueDim) = Ws_baseline;

  // h1 = ReLU(W_l1 @ h0 + b_l1)
  Eigen::Array<float, kBackupLayer1Dim, 1> h1_pre = (W_l1_ * h0).array() + b_l1_;
  Eigen::Matrix<float, kBackupLayer1Dim, 1> h1 = h1_pre.max(0.0f).matrix();

  // h2 = ReLU(W_l2 @ h1 + b_l2)
  Eigen::Array<float, kBackupLayer2Dim, 1> h2_pre = (W_l2_ * h1).array() + b_l2_;
  Eigen::Matrix<float, kBackupLayer2Dim, 1> h2 = h2_pre.max(0.0f).matrix();

  // out (residual) = W_out @ h2 + b_out
  Eigen::Array<float, kBackupOutputDim, 1> out = (W_out_ * h2).array() + b_out_;

  // S/W-skip ("AlphaZero passthrough"): see py/shared/backup_net.py for the design and the
  // matching Python implementation. At init the residual is zero, so the logits are exactly
  // log(clamp(S_baseline)) and W = Ws_baseline; softmax then recovers S_baseline to within the
  // clamp tolerance. Training fits the residual.
  for (int i = 0; i < kValueDim; ++i) {
    out(i) += std::log(std::max(S_baseline(i), detail::kSstarClampEps));
  }
  out(kValueDim) += Ws_baseline;

  // Softmax the logits to obtain the active-seat-rotated S distribution.
  Tensor S_rotated;
  for (int i = 0; i < kValueDim; ++i) S_rotated(i) = out(i);
  eigen_util::softmax_in_place(S_rotated);

  ActiveSeatResult result;
  result.S = S_rotated;
  result.W = std::max(out(kValueDim), 1e-4f);
  return result;
}

}  // namespace beta0
