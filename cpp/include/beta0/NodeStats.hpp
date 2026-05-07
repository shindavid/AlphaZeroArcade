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
  int RN = 0;       // real count
  int VN = 0;       // virtual count

  // TODO: generalize these fields to utility lower/upper bounds
  player_bitset_t provably_winning;
  player_bitset_t provably_losing;

  // Running backup-NN accumulator: backup_accu_static + sum of W_child @ [N_i, Q_i, W_i] over
  // children. Incrementally maintained: on each visit, the changed child's old contribution is
  // subtracted and the new one is added.
  AccumulatorArray backup_accumulator;
};

}  // namespace beta0
