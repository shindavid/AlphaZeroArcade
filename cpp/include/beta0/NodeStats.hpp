#pragma once

#include "beta0/SpecTraits.hpp"
#include "beta0/concepts/SpecConcept.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
struct NodeStats {
  using Game = Spec::Game;
  using ValueArray = Game::Types::ValueArray;
  using player_bitset_t = Game::Types::player_bitset_t;
  using AccumulatorArray = SpecTraits<Spec>::AccumulatorArray;

  int total_count() const { return RN + VN; }

  ValueArray Q;     // excludes virtual loss
  ValueArray Q_sq;  // excludes virtual loss
  ValueArray W;     // uncertainty estimate, per-player

  // Qs_star/Ws_star: the baselines that BackupNet sees as context inputs (alongside the
  // accumulator and the static latent z_s). Computed by update_stats() as the prior-augmented
  // children average (LoTE for Qs_star, LoTV for Ws_star), and preserved here even when the
  // backup-NN override replaces stats.Q / stats.W. The trailing 's' stands for 'seat': only
  // the active-seat scalar is recorded (not a per-player ValueArray). The names match the
  // math notation in docs/BetaZero.pdf and the Python BackupNet's `Qs_star`, `Ws_star`
  // arguments.
  float Qs_star = 0.0f;
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
