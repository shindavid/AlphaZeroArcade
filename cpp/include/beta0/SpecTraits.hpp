#pragma once

#include "beta0/concepts/SpecConcept.hpp"

#include <Eigen/Core>

namespace beta0 {

/*
 * SpecTraits<Spec> centralizes the type aliases that depend on Spec's compile-time constants.
 *
 * Putting them here prevents repetition across Node data structures, BackupNNEvaluator, and the
 * Manager, and ensures consistent sizing everywhere.
 */
template <beta0::concepts::Spec Spec>
struct SpecTraits {
  using Game = Spec::Game;
  using TensorEncodings = Spec::TensorEncodings;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using ValueArray = Game::Traits::ValueArray;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  // BackupNet dimensions (forwarded from Spec::BackupNetDims, which mirrors the Python spec).
  using BackupNetDims = Spec::BackupNetDims;
  static constexpr int kStaticLatentDim = BackupNetDims::kStaticLatentDim;
  static constexpr int kEmbedDim = BackupNetDims::kEmbedDim;
  static constexpr int kBackupLayer1Dim = BackupNetDims::kBackupLayer1Dim;
  static constexpr int kBackupLayer2Dim = BackupNetDims::kBackupLayer2Dim;
  static constexpr int kZaDim = BackupNetDims::kZaDim;

  // Per-action child-stats vector layout: [Qs, Ws, N, P, AVs, AUs].
  static constexpr int kChildStatDim = 6;

  // Width of the value head's logit vector (e.g. 3 for WLD, 2 for WL).
  static constexpr int kValueDim = GameResultEncoding::Tensor::Dimensions::total_size;

  // ChildEmbeddingHead input width.
  static constexpr int kPerChildInDim = kChildStatDim + kZaDim;

  // BackupNet layer-1 input width: [accumulator; z_s; Ss*; Ws*]. Ss* is a value_dim-vector
  // (the active-seat-rotated WLD/WL distribution baseline); Ws* stays a scalar.
  static constexpr int kBackupLayer1InDim = kEmbedDim + kStaticLatentDim + kValueDim + 1;

  // BackupNet output width: [Q logits...; W scalar].
  static constexpr int kBackupOutputDim = kValueDim + 1;

  // Fixed-size Eigen column-array for the backup-NN accumulator (sum of per-child embeddings).
  // Stored on each node; updated incrementally during search.
  using AccumulatorArray = Eigen::Array<float, kEmbedDim, 1>;

  // Per-edge cached embedding e_i, used for NNUE-style subtract-add updates.
  using EmbedArray = Eigen::Array<float, kEmbedDim, 1>;

  // Per-action latent z_a (cached on each Edge at parent-evaluation time).
  using ZaArray = Eigen::Array<float, kZaDim, 1>;

  // Per-node static latent z_s.
  using StaticLatentArray = Eigen::Array<float, kStaticLatentDim, 1>;
};

}  // namespace beta0
