#include "beta0/BackupNNEvaluator.hpp"

#include "util/Asserts.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
void BackupNNEvaluator<Spec>::add_child_contribution(int N, const ValueArray& Q,
                                                     const ValueArray& W,
                                                     AccumulatorArray& acc) const {
  // Build row-vector input: [N, Q_0, ..., Q_{P-1}, W_0, ..., W_{P-1}]
  using InputVec = Eigen::Matrix<float, 1, kChildInputDim, Eigen::RowMajor>;
  InputVec x;
  x(0, 0) = static_cast<float>(N);
  for (int p = 0; p < kNumPlayers; ++p) {
    x(0, 1 + p) = Q(p);
    x(0, 1 + kNumPlayers + p) = W(p);
  }

  // acc (as row-vec) += x @ W_child_  ([1, kChildInputDim] @ [kChildInputDim, kBackupHiddenDim])
  using AccVec = Eigen::Matrix<float, 1, kBackupHiddenDim, Eigen::RowMajor>;
  Eigen::Map<AccVec> acc_map(acc.data());
  acc_map.noalias() += x * W_child_;
}

template <beta0::concepts::Spec Spec>
typename BackupNNEvaluator<Spec>::QWPair BackupNNEvaluator<Spec>::apply(
  const AccumulatorArray& acc) const {
  using AccVec = Eigen::Matrix<float, 1, kBackupHiddenDim, Eigen::RowMajor>;
  using OutVec = Eigen::Matrix<float, 1, kOutputDim, Eigen::RowMajor>;

  // h = ReLU(acc)
  Eigen::Map<const AccVec> acc_map(acc.data());
  AccVec h = acc_map.cwiseMax(0.0f);

  // out = h @ W_out_ + b_out_  (b_out_ viewed as row vector)
  Eigen::Map<const Eigen::Matrix<float, 1, kOutputDim, Eigen::RowMajor>> b_row(b_out_.data());
  OutVec out = h * W_out_ + b_row;

  // Split into Q and W per player
  ValueArray Q_out, W_out_arr;
  for (int p = 0; p < kNumPlayers; ++p) {
    Q_out(p) = out(0, p);
    W_out_arr(p) = out(0, kNumPlayers + p);
  }
  return {Q_out, W_out_arr};
}

template <beta0::concepts::Spec Spec>
void BackupNNEvaluator<Spec>::load(const float* weights, size_t n_floats) {
  RELEASE_ASSERT(n_floats == kWeightCount, "BackupNNEvaluator::load: expected {} floats, got {}",
                 kWeightCount, n_floats);

  const float* ptr = weights;

  // W_child_: [kChildInputDim, kBackupHiddenDim] row-major
  std::copy_n(ptr, W_child_.size(), W_child_.data());
  ptr += W_child_.size();

  // W_out_: [kBackupHiddenDim, kOutputDim] row-major
  std::copy_n(ptr, W_out_.size(), W_out_.data());
  ptr += W_out_.size();

  // b_out_: [kOutputDim]
  std::copy_n(ptr, b_out_.size(), b_out_.data());

  ready_ = true;
}

}  // namespace beta0
