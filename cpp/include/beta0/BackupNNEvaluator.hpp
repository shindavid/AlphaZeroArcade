#pragma once

#include "beta0/SpecTraits.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "core/LoopControllerListener.hpp"
#include "core/ReceivedModel.hpp"

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
 *     h0   = [acc ; z_s ; Qs* ; Ws*]                                       (BackupNet input)
 *     h1   = ReLU(W_l1 @ h0 + b_l1)
 *     h2   = ReLU(W_l2 @ h1 + b_l2)
 *     out  = W_out @ h2 + b_out                                            (kValueDim+1,)
 *
 * out[0:kValueDim] are Q logits in the same format as the base-NN's value head (e.g. WLD logits
 * for c4); out[kValueDim] is the W (uncertainty) scalar. apply() softmax-collapses the Q logits
 * into an active-seat win-share scalar via GameResultEncoding::to_value_array.
 *
 * Weights arrive over the wire as orphan ONNX initializers under nnue/ in the model file. The
 * loop-controller pushes a ReceivedModel for every weight reload; this class is registered as a
 * kReloadWeights listener by Manager during construction.
 */
template <beta0::concepts::Spec Spec>
class BackupNNEvaluator
    : public core::LoopControllerListener<core::LoopControllerInteractionType::kReloadWeights> {
 public:
  using Traits = SpecTraits<Spec>;
  using GameResultEncoding = Traits::GameResultEncoding;
  using ValueArray = Traits::ValueArray;
  using AccumulatorArray = Traits::AccumulatorArray;
  using EmbedArray = Traits::EmbedArray;
  using ZaArray = Traits::ZaArray;
  using StaticLatentArray = Traits::StaticLatentArray;

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
                "BackupNNEvaluator currently assumes 2-player zero-sum (active-seat scalar Q).");

  // Per-action child-stats vector: [Qs, Ws, N, P, AVs, AUs].
  using ChildStatArray = Eigen::Array<float, kChildStatDim, 1>;

  // apply()'s return type: active-seat scalar Q (win-share) and scalar W.
  struct ActiveSeatQW {
    float Q;
    float W;
  };

  bool ready() const { return ready_; }

  // Reload weights from the loop-controller. Looks up nnue/{child_embed,layer1,layer2,out}.
  // {weight,bias} in `model.nnue_weights` and copies them into the matrices below; asserts each
  // tensor has exactly the expected number of floats. Sets ready_ = true.
  void reload_weights(const core::ReceivedModel& model) override;

  // Compute one child embedding e_i = ReLU(W_e @ [cs ; za] + b_e) * (cs(P_INDEX) > 0).
  EmbedArray compute_child_embedding(const ChildStatArray& cs, const ZaArray& za) const;

  // Apply BackupNet to (acc, z_s, Qs*, Ws*) and return (Q_active_winshare, W_scalar).
  ActiveSeatQW apply(const AccumulatorArray& acc, const StaticLatentArray& z_s,
                     float Qs_star, float Ws_star) const;

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
