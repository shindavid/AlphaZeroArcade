#pragma once

#include "beta0/SpecTraits.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "search/EdgeBase.hpp"

namespace beta0 {

/*
 * An Edge corresponds to an action that can be taken from this node.
 *
 * Extends alpha0::Edge with:
 *   - child_AU: per-player action-value uncertainty (set at parent eval time)
 *   - action_latent:      per-action latent vector consumed by ChildEmbeddingHead (set at parent eval
 *               time from the action_latent network head)
 *   - e_cached: most recent value of e_i = ReLU(W_e @ [child_stats; action_latent] + b_e) * (P>0).
 *               Used by BackupNNEvaluator for NNUE-style subtract-add updates of the parent's
 *               backup_accumulator. Initialized at parent expansion to e_i evaluated with
 *               (Qs=0, Ws=0, N=0).
 */
template <beta0::concepts::Spec Spec>
struct Edge : public search::EdgeBase<typename Spec::Game> {
  using Traits = SpecTraits<Spec>;
  using ValueArray = Spec::Game::Traits::ValueArray;
  using ActionLatentArray = Traits::ActionLatentArray;
  using EmbedArray = Traits::EmbedArray;

  Edge() {
    child_AV.fill(0);
    child_AU.fill(0);
    action_latent.setZero();
    e_cached.setZero();
  }

  // The child's NodeStats::backprop_counter at the time e_cached was last refreshed. Compared
  // against the child's current counter inside parent's update_stats() to detect whether this
  // edge's contribution to the parent's backup_accumulator is stale and needs subtract-add.
  // Sentinel -1 means "no e_cached has been computed against any child state yet" (the
  // expansion seed sets this to 0 to match a freshly-constructed child whose counter is 0).
  int last_seen_child_counter = -1;

  int E = 0;  // real or virtual count
  float policy_prior_prob = 0;

  // policy_prior_prob + adjustments (from Dirichlet-noise and softmax-temperature)
  float adjusted_base_prob = 0;

  // child_AV is set with the neural network's AV estimate of this edge's action at the time the
  // parent is evaluated.
  ValueArray child_AV;

  // child_AU is set with the neural network's AU (action-value uncertainty) estimate at the time
  // the parent is evaluated.
  ValueArray child_AU;

  // Per-action latent action_latent (consumed by BackupNNEvaluator::compute_child_embedding). Set at
  // parent-evaluation time from the action_latent network head.
  ActionLatentArray action_latent;

  // Cached value of this edge's contribution e_i to its parent's backup_accumulator. Mutated
  // by BackupNNEvaluator::add_child_contribution / Manager::update_stats so the parent's
  // accumulator can be maintained incrementally (NNUE-style subtract-add).
  EmbedArray e_cached;
};

}  // namespace beta0
