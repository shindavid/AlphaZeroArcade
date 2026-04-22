#include "beta0/APhiEvaluator.hpp"

#include "util/Asserts.hpp"

#include <algorithm>
#include <cstring>

namespace beta0 {

template <beta0::concepts::Spec Spec>
void APhiEvaluator<Spec>::add_child_contribution(int N, const ValueArray& Q, const ValueArray& W,
                                                  AccumulatorArray& acc) const {
  // Build input vector: [N, Q_0, ..., Q_{P-1}, W_0, ..., W_{P-1}]
  using InputVec = Eigen::Matrix<float, 1, kChildInputDim, Eigen::RowMajor>;
  InputVec x;
  x(0, 0) = static_cast<float>(N);
  for (int p = 0; p < kNumPlayers; ++p) {
    x(0, 1 + p) = Q(p);
    x(0, 1 + kNumPlayers + p) = W(p);
  }

  // acc += x @ W_AD   (shape: [1, kChildInputDim] @ [kChildInputDim, kHiddenDim])
  using WAD = Eigen::Matrix<float, kChildInputDim, kHiddenDim, Eigen::RowMajor>;
  using AccVec = Eigen::Matrix<float, 1, kHiddenDim, Eigen::RowMajor>;
  Eigen::Map<const WAD> W_ad(W_AD_.data());
  Eigen::Map<AccVec> acc_map(acc.data());
  acc_map.noalias() += x * W_ad;
}

template <beta0::concepts::Spec Spec>
std::pair<typename APhiEvaluator<Spec>::ValueArray,
          typename APhiEvaluator<Spec>::ValueArray>
APhiEvaluator<Spec>::apply(const AccumulatorArray& acc) const {
  using AccVec = Eigen::Matrix<float, 1, kHiddenDim, Eigen::RowMajor>;
  using WOut = Eigen::Matrix<float, kHiddenDim, kOutputDim, Eigen::RowMajor>;
  using OutVec = Eigen::Matrix<float, 1, kOutputDim, Eigen::RowMajor>;

  // h = ReLU(acc)
  Eigen::Map<const AccVec> acc_map(acc.data());
  AccVec h = acc_map.cwiseMax(0.0f);

  // out = h @ W_out + b_out
  Eigen::Map<const WOut> W_out(W_out_.data());
  Eigen::Map<const OutVec> b_out(b_out_.data());
  OutVec out = h * W_out + b_out;

  // Split into Q and W per player
  ValueArray Q_out, W_out_arr;
  for (int p = 0; p < kNumPlayers; ++p) {
    Q_out(p) = out(0, p);
    W_out_arr(p) = out(0, kNumPlayers + p);
  }
  return {Q_out, W_out_arr};
}

template <beta0::concepts::Spec Spec>
void APhiEvaluator<Spec>::load(const float* weights, size_t n_floats) {
  RELEASE_ASSERT(n_floats == kWeightCount,
                 "APhiEvaluator::load: expected {} floats, got {}", kWeightCount, n_floats);

  const float* ptr = weights;
  std::copy_n(ptr, W_AD_.size(), W_AD_.data());
  ptr += W_AD_.size();
  std::copy_n(ptr, W_out_.size(), W_out_.data());
  ptr += W_out_.size();
  std::copy_n(ptr, b_out_.size(), b_out_.data());

  ready_ = true;
}

}  // namespace beta0
