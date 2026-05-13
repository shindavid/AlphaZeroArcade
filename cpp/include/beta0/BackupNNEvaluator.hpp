#pragma once

#include "beta0/SpecTraits.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "core/AuxEvalService.hpp"
#include "core/ModelBundle.hpp"

#include <Eigen/Dense>

namespace beta0 {

/*
 * BackupNNEvaluator implements the BetaZero CPU-side BackupNet. See docs/BetaZero.pdf
 * (Section 4.3) and py/shared/backup_net.py for the design.
 *
 * Architecture (all dense, ReLU activations):
 *
 *     e_i  = ReLU(W_child_embed @ [child_stats_i ; z_a,i] + b_child_embed) * (P_i > 0)
 *     acc  = sum_i e_i                                                     (NNUE accumulator)
 *     h0   = [acc ; z_s ; Ss* ; Ws*]                                       (BackupNet input)
 *     h1   = ReLU(W_l1 @ h0 + b_l1)
 *     h2   = ReLU(W_l2 @ h1 + b_l2)
 *     dOut = W_out @ h2 + b_out                                            (kValueDim+1,)
 *
 *     out[0:kValueDim] = dOut[0:kValueDim] + log(clamp(Ss*))               (S-skip)
 *     out[kValueDim]   = dOut[kValueDim]   + Ws*                           (W-skip)
 *
 * out[0:kValueDim] are logits in the same WLD/WL format as the base-NN's value head, in the
 * active-seat frame; out[kValueDim] is the W (uncertainty) scalar. apply() softmax-collapses
 * the logits into an active-seat-rotated S Tensor (the full distribution, not just the
 * win-share scalar), so the caller can write a calibrated S into NodeStats and let
 * NodeStats::Q() derive the per-player win-share view.
 *
 * S-skip (BetaZero "AlphaZero passthrough"): the MLP head's weights and biases are
 * zero-initialized on the Python side, so at gen 0 the residual dOut is zero and apply()
 * returns exactly Ss*. Training only fits the residual. The skip clamp constant
 * (kSstarClampEps) MUST match the Python side byte-for-byte; the equivalence unit test
 * verifies this.
 *
 * Weights arrive over the wire as orphan ONNX initializers under nnue/ in the model file.
 * BackupNNEvaluator is constructed (unloaded) by the AuxFactory wired into NNEvaluationService
 * at service creation time. The owning NNEvaluationService drives this evaluator's
 * reload_weights() -- once at startup if a local model file was provided, and once for every
 * subsequent loop-controller-pushed reload.
 *
 * NNUE invariant and the N=0 question
 * -----------------------------------
 * The subtract-add chain that NNUE relies on flows through `S_baseline` / `W_baseline`, NOT
 * through `stats.S` / `stats.W`. That is: a parent's `backup_accumulator` is built from
 *
 *     e_i = compute_child_embedding(child_i.S_baseline, child_i.W_baseline, ...)
 *
 * and `S_baseline`/`W_baseline` are pure LoTE/LoTV running averages -- they are captured in
 * update_stats() *before* the apply() override is written into `stats.S` (see Manager.inl).
 * `stats.S` (the apply() output) is a leaf of the NNUE computation: it is read by PUCT and
 * by final search results, but never feeds back into another node's accumulator. So the
 * subtract-add invariant holds regardless of what `stats.S` reports at any node.
 *
 * This means we have a free design choice for `stats.S` at N=0 (i.e., immediately after
 * expansion, before any backprop has occurred). We choose to leave it equal to R (the V-head
 * output), exactly as load_evaluations() seeds it -- apply() is simply not invoked at
 * expansion time. Equivalently, you can think of this as an implicit step gate g(N) = [N>0]
 * on apply()'s residual: report R when there is no search evidence, and apply() once at
 * least one backprop has accumulated evidence.
 *
 * This choice is preferable to invoking apply() at N=0 because V is trained directly against
 * the game outcome Z, while apply()'s output at N=0 has no training signal anchoring it (the
 * BackupNet is trained only at n in {1, ..., N} -- see docs/BetaZero.pdf, Section 6.1). The
 * V-head is therefore the better N=0 estimate, and routing it through apply() would only
 * introduce an unconstrained residual without buying anything.
 */
template <beta0::concepts::Spec Spec>
class BackupNNEvaluator : public core::AuxEvalService {
 public:
  using Traits = SpecTraits<Spec>;
  using GameResultEncoding = Traits::GameResultEncoding;
  using ValueArray = Traits::ValueArray;
  using AccumulatorArray = Traits::AccumulatorArray;
  using EmbedArray = Traits::EmbedArray;
  using ZaArray = Traits::ZaArray;
  using StaticLatentArray = Traits::StaticLatentArray;
  using Tensor = GameResultEncoding::Tensor;

  static constexpr int kNumPlayers = Traits::kNumPlayers;
  static constexpr int kStaticLatentDim = Traits::kStaticLatentDim;
  static constexpr int kEmbedDim = Traits::kEmbedDim;
  static constexpr int kBackupLayer1Dim = Traits::kBackupLayer1Dim;
  static constexpr int kBackupLayer2Dim = Traits::kBackupLayer2Dim;
  static constexpr int kZaDim = Traits::kZaDim;
  static constexpr int kChildStatDim = Traits::kChildStatDim;
  static constexpr int kPerChildInDim = Traits::kPerChildInDim;
  static constexpr int kBackupLayer1InDim = Traits::kBackupLayer1InDim;
  static constexpr int kValueDim = Traits::kValueDim;
  static constexpr int kBackupOutputDim = Traits::kBackupOutputDim;

  static_assert(kNumPlayers == 2,
                "BackupNNEvaluator currently assumes 2-player zero-sum.");

  // Per-action child-stats vector: [Qs, Ws, N, P, AVs, AUs].
  using ChildStatArray = Eigen::Array<float, kChildStatDim, 1>;

  // apply()'s return type: active-seat-rotated S Tensor (full WLD/WL distribution) and
  // scalar W. The caller is responsible for un-rotating S back to canonical frame before
  // storing in NodeStats::S.
  struct ActiveSeatResult {
    Tensor S;
    float W;
  };

  // Construct an unloaded evaluator. The owning NNEvaluationService is responsible for
  // calling reload_weights() before any apply() / compute_child_embedding() call.
  BackupNNEvaluator() = default;

  bool ready() const { return ready_; }

  // Reload weights from the loop-controller. Looks up nnue/{child_embed,layer1,layer2,out}.
  // {weight,bias} in `model.nnue_weights` and copies them into the matrices below; asserts each
  // tensor has exactly the expected number of floats. Sets ready_ = true.
  void reload_weights(const core::ModelBundle& model) override;

  // Compute one child embedding e_i = ReLU(W_e @ [cs ; za] + b_e) * (cs(P_INDEX) > 0).
  EmbedArray compute_child_embedding(const ChildStatArray& cs, const ZaArray& za) const;

  // Apply BackupNet to (acc, z_s, Ss*, Ws*) and return (S_active_seat_rotated, W_scalar).
  ActiveSeatResult apply(const AccumulatorArray& acc, const StaticLatentArray& z_s,
                         const Tensor& S_baseline, float Ws_baseline) const;

 private:
  // Index of P in the per-action child-stats vector [Qs, Ws, N, P, AVs, AUs].
  static constexpr int kPolicyPriorIndex = 3;

  // ChildEmbeddingHead.child_embed: (kEmbedDim, kPerChildInDim) row-major.
  Eigen::Matrix<float, kEmbedDim, kPerChildInDim, Eigen::RowMajor> W_child_embed_;
  Eigen::Array<float, kEmbedDim, 1> b_child_embed_;

  // BackupNet.layer1: (kBackupLayer1Dim, kBackupLayer1InDim) row-major.
  Eigen::Matrix<float, kBackupLayer1Dim, kBackupLayer1InDim, Eigen::RowMajor> W_l1_;
  Eigen::Array<float, kBackupLayer1Dim, 1> b_l1_;

  // BackupNet.layer2: (kBackupLayer2Dim, kBackupLayer1Dim) row-major.
  Eigen::Matrix<float, kBackupLayer2Dim, kBackupLayer1Dim, Eigen::RowMajor> W_l2_;
  Eigen::Array<float, kBackupLayer2Dim, 1> b_l2_;

  // BackupNet.out: (kBackupOutputDim, kBackupLayer2Dim) row-major.
  Eigen::Matrix<float, kBackupOutputDim, kBackupLayer2Dim, Eigen::RowMajor> W_out_;
  Eigen::Array<float, kBackupOutputDim, 1> b_out_;

  bool ready_ = false;
};

}  // namespace beta0

#include "inline/beta0/BackupNNEvaluator.inl"
