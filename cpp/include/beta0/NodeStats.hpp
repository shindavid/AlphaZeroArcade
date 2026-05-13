#pragma once

#include "beta0/SpecTraits.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "core/BasicTypes.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
struct NodeStats {
  using Game = Spec::Game;
  using Traits = SpecTraits<Spec>;
  using GameResultEncoding = Traits::GameResultEncoding;
  using Tensor = GameResultEncoding::Tensor;  // canonical (per-seat-0) WLD/WL distribution
  using ValueArray = Game::Traits::ValueArray;
  using player_bitset_t = Game::Traits::player_bitset_t;
  using AccumulatorArray = Traits::AccumulatorArray;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  int total_count() const { return RN + VN; }

  // ValueArray "Q" is just the win-share view of the canonical S distribution. Mirrors the
  // alpha0::NodeStableData R (field) / V() (accessor) convention. The seat overload returns
  // the per-seat scalar directly, so callers can write `stats.Q(seat)` cleanly.
  ValueArray Q() const { return GameResultEncoding::to_value_array(S); }
  float Q(core::seat_index_t seat) const { return Q()(seat); }

  void setW(float w) { W.setConstant(w); }

  Tensor S;      // canonical posterior result distribution (sums to 1); excludes virtual loss
  ValueArray W;  // uncertainty estimate, per-player

  // S_baseline / W_baseline: the prior-augmented children-average baselines (LoTE for S_baseline,
  // LoTV for W_baseline) that BackupNet sees as context inputs (alongside the accumulator and the
  // static latent z_s). Computed by update_stats() and preserved here even when the backup-NN
  // override replaces stats.S / stats.W.
  //
  // The NNUE subtract-add chain runs through S_baseline / W_baseline, NOT through S / W. That
  // is, a parent's backup_accumulator sums per-child embeddings computed from each child's
  // S_baseline / W_baseline, never from the BackupNet-overridden S / W. See
  // BackupNNEvaluator.hpp ("NNUE invariant and the N=0 question") for the implications,
  // including why we leave stats.S = R at expansion (N=0) rather than invoking apply().
  //
  // Stored in the canonical (un-rotated) frame, mirroring the storage convention for stats.S
  // and stats.W. The active-seat-rotated view is materialized on demand at the BackupNet
  // call site (and at training-target encoding time) via GameResultEncoding::left_rotate /
  // ValueArray indexing by the child's active seat.
  Tensor S_baseline;
  ValueArray W_baseline = ValueArray::Zero();
  int RN = 0;  // real count
  int VN = 0;  // virtual count

  // Monotonically incremented (under the node's mutex) once per backprop() call. Read by
  // parents during their own update_stats() to detect which child contributions have changed
  // since the last incremental backup-NN accumulator update. Each Edge caches the
  // backprop_counter value seen the last time its e_cached was refreshed.
  int backprop_counter = 0;

  // TODO: generalize these fields to utility lower/upper bounds
  player_bitset_t provably_winning;
  player_bitset_t provably_losing;

  // Running backup-NN accumulator: sum_i e_i over child edges, where each e_i is the per-child
  // embedding produced by ChildEmbeddingHead (= ReLU(W_e @ [child_stats_i; z_a,i] + b_e), masked
  // by P_i > 0). Seeded at expansion with child_stats = (Qs=0, Ws=0, N=0, P, AVs, AUs); maintained
  // incrementally during backups via NNUE-style subtract-add against Edge::e_cached.
  AccumulatorArray backup_accumulator;
};

}  // namespace beta0
