#pragma once

#include "beta0/SpecTraits.hpp"
#include "beta0/concepts/SpecConcept.hpp"

#include <Eigen/Dense>

namespace beta0 {

/*
 * BackupNNEvaluator implements the backup neural network for BetaZero.
 *
 * Architecture (1-hidden-layer NNUE MLP):
 *
 *   backup_accumulator = backup_accu_static + sum_i W_child @ [N_i, Q_i..., W_i...]
 *   h                  = ReLU(backup_accumulator)
 *   [Q_out, W_out]     = W_out @ h + b_out          (outputs 2*kNumPlayers scalars)
 *
 * backup_accu_static is precomputed on GPU by the main NN (head "backup_accu_static") and
 * loaded into NodeStableData. The per-child dynamic contribution W_child @ [N_i, Q_i, W_i] is
 * accumulated CPU-side in update_stats() on each visit.
 *
 * Weight layout (flat float array passed to load()):
 *   [W_child: kChildInputDim * kBackupHiddenDim]  row-major [kChildInputDim, kBackupHiddenDim]
 *   [W_out:   kBackupHiddenDim * kOutputDim]      row-major [kBackupHiddenDim, kOutputDim]
 *   [b_out:   kOutputDim]
 *
 * where kChildInputDim = 1 + 2*kNumPlayers  ([N, Q_0..Q_{P-1}, W_0..W_{P-1}])
 *       kOutputDim     = 2 * kNumPlayers    ([Q_parent_0..., W_parent_0...])
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
