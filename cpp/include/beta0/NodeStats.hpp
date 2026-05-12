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
  // alpha0::NodeStableData R (field) / V() (accessor) convention.
  ValueArray Q() const { return GameResultEncoding::to_value_array(S); }

  // Set S to a degenerate distribution that has the active seat winning with probability `q`
  // and the other seat winning with probability `1 - q` (zero draw mass). Used at terminal
  // and provably-resolved nodes.
  void set_S_from_active_winshare(float q, core::seat_index_t seat) {
    S.setZero();
    S(seat) = q;
    S(1 - seat) = 1.0f - q;
  }

  void setW(float w) { W.setConstant(w); }

  Tensor S;       // canonical posterior result distribution (sums to 1); excludes virtual loss
  ValueArray W;   // uncertainty estimate, per-player

  // Ss_star/Ws_star: the baselines that BackupNet sees as context inputs (alongside the
  // accumulator and the static latent z_s). Computed by update_stats() as the prior-augmented
  // children average (LoTE for Ss_star, LoTV for Ws_star), and preserved here even when the
  // backup-NN override replaces stats.S / stats.W.
  //
  // Ss_star is the active-seat-rotated view of the LoTE-averaged S, so Ss_star(0) is always
  // the active seat's "win" mass (parallels how Qs_star used to be the active-seat scalar).
  // Ws_star remains a scalar (just the active-seat W).
  //
  // The names match the math notation in docs/BetaZero.pdf and the Python BackupNet's
  // `Ss_star`, `Ws_star` arguments.
  Tensor Ss_star;
  float Ws_star = 0.0f;
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
