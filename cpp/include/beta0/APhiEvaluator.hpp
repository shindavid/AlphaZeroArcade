#pragma once

#include "beta0/concepts/SpecConcept.hpp"
#include "util/EigenUtil.hpp"

#include <Eigen/Dense>

#include <array>
#include <utility>
#include <vector>

namespace beta0 {

/*
 * APhiEvaluator implements the A_phi "backup network" for BetaZero.
 *
 * Architecture (1-hidden-layer NNUE MLP):
 *
 *   phi_accumulator = phi_accu_static + sum_i W_AD @ [N_i, Q_i..., W_i...]
 *   h               = ReLU(phi_accumulator)
 *   [Q_out, W_out]  = h @ W_out + b_out          (outputs 2*kNumPlayers scalars)
 *
 * phi_accu_static is precomputed on GPU by the main NN (head "phi_accu_static") and loaded into
 * NodeStableData. The per-child dynamic contribution W_AD @ [N_i, Q_i, W_i] is accumulated
 * CPU-side in update_stats() each visit.
 *
 * Weight layout (flat float array passed to load()):
 *   [W_AD: kChildInputDim * kHiddenDim]   row-major [kChildInputDim, kHiddenDim]
 *   [W_out: kHiddenDim * kOutputDim]      row-major [kHiddenDim, kOutputDim]
 *   [b_out: kOutputDim]
 *
 * where kChildInputDim = 1 + 2*kNumPlayers  ([N, Q_0..Q_{P-1}, W_0..W_{P-1}])
 *       kOutputDim     = 2 * kNumPlayers    ([Q_parent_0..., W_parent_0...])
 */
template <beta0::concepts::Spec Spec>
class APhiEvaluator {
 public:
  using Game = Spec::Game;
  using ValueArray = Game::Types::ValueArray;

  static constexpr int kHiddenDim = Spec::kPhiHiddenDim;
  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kChildInputDim = 1 + 2 * kNumPlayers;
  static constexpr int kOutputDim = 2 * kNumPlayers;
  static constexpr size_t kWeightCount =
      kChildInputDim * kHiddenDim + kHiddenDim * kOutputDim + kOutputDim;

  using AccumulatorArray = std::array<float, kHiddenDim>;

  bool ready() const { return ready_; }

  // Adds the contribution of one child to the accumulator:
  //   acc += W_AD @ [N, Q_0, ..., Q_{P-1}, W_0, ..., W_{P-1}]
  void add_child_contribution(int N, const ValueArray& Q, const ValueArray& W,
                              AccumulatorArray& acc) const;

  // Evaluates: h = ReLU(acc);  [Q_out, W_out] = h @ W_out_ + b_out_
  // Returns (Q_parent, W_parent) as a pair of ValueArrays.
  std::pair<ValueArray, ValueArray> apply(const AccumulatorArray& acc) const;

  // Loads weights from a flat float array of exactly kWeightCount elements.
  void load(const float* weights, size_t n_floats);

 private:
  // W_AD: [kChildInputDim, kHiddenDim] row-major
  std::array<float, kChildInputDim * kHiddenDim> W_AD_{};
  // W_out: [kHiddenDim, kOutputDim] row-major
  std::array<float, kHiddenDim * kOutputDim> W_out_{};
  // b_out: [kOutputDim]
  std::array<float, kOutputDim> b_out_{};

  bool ready_ = false;
};

}  // namespace beta0

#include "inline/beta0/APhiEvaluator.inl"
