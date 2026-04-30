#pragma once

#include "beta0/SpecTraits.hpp"
#include "beta0/concepts/SpecConcept.hpp"

#include <Eigen/Dense>

namespace beta0 {

/*
 * BackupNNEvaluator implements the backup neural network for BetaZero.
 *
 * TODO: the details on the dimensions here are out-of-date. Revise this based on the latest beta0
 * design document.
 *
 * TODO: this should extend core::LoopControllerListener<kReloadWeights>, and get the nnue-weights
 * from the passed-in ReceivedModel struct.
 */
template <beta0::concepts::Spec Spec>
class BackupNNEvaluator {
 public:
  using Traits = SpecTraits<Spec>;
  using ValueArray = Traits::ValueArray;
  using AccumulatorArray = Traits::AccumulatorArray;
  using QWPair = Traits::QWPair;

  static constexpr int kBackupHiddenDim = Spec::kBackupHiddenDim;
  static constexpr int kNumPlayers = Traits::kNumPlayers;
  static constexpr int kChildInputDim = 1 + 2 * kNumPlayers;
  static constexpr int kOutputDim = 2 * kNumPlayers;
  static constexpr size_t kWeightCount =
    kChildInputDim * kBackupHiddenDim + kBackupHiddenDim * kOutputDim + kOutputDim;

  bool ready() const { return ready_; }

  // Adds one child's contribution to the accumulator:
  //   acc += W_child.T @ [N, Q_0, ..., Q_{P-1}, W_0, ..., W_{P-1}]
  void add_child_contribution(int N, const ValueArray& Q, const ValueArray& W,
                              AccumulatorArray& acc) const;

  // Evaluates: h = ReLU(acc);  [Q_out, W_out] = W_out_ @ h + b_out_
  // Returns a QWPair with predicted (Q_parent, W_parent).
  QWPair apply(const AccumulatorArray& acc) const;

  // Loads weights from a flat float array of exactly kWeightCount elements.
  void load(const float* weights, size_t n_floats);

 private:
  // W_child: [kChildInputDim, kBackupHiddenDim] row-major
  Eigen::Matrix<float, kChildInputDim, kBackupHiddenDim, Eigen::RowMajor> W_child_;
  // W_out: [kBackupHiddenDim, kOutputDim] row-major
  Eigen::Matrix<float, kBackupHiddenDim, kOutputDim, Eigen::RowMajor> W_out_;
  // b_out: [kOutputDim]
  Eigen::Array<float, kOutputDim, 1> b_out_;

  bool ready_ = false;
};

}  // namespace beta0

#include "inline/beta0/BackupNNEvaluator.inl"
