#pragma once

#include "alpha0/NodeStableData.hpp"
#include "beta0/SpecTraits.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "core/BasicTypes.hpp"

namespace beta0 {

/*
 * Extends alpha0::NodeStableData with an uncertainty field U (prior W), populated from the
 * "uncertainty" network head.
 */
template <beta0::concepts::Spec Spec>
struct NodeStableData : public alpha0::NodeStableData<Spec> {
  using Base = alpha0::NodeStableData<Spec>;
  using Game = Spec::Game;
  using State = Game::State;
  using TensorEncodings = Spec::TensorEncodings;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using GameOutcome = Game::Traits::GameOutcome;
  using ValueArray = Game::Traits::ValueArray;
  using AccumulatorArray = SpecTraits<Spec>::AccumulatorArray;
  using StaticLatentArray = SpecTraits<Spec>::StaticLatentArray;

  // WinShareTensor = FTensor<Sizes<kNumPlayers>> — per-player uncertainties
  using WinShareTensor = TensorEncodings::WinShareTensor;

  NodeStableData(const State&, int n_valid_moves, core::seat_index_t);
  NodeStableData(const State&, const GameOutcome&);

  // Returns the prior uncertainty as a ValueArray (U from the neural network).
  ValueArray U() const { return Eigen::Map<const ValueArray>(uncertainty_.data()); }

  WinShareTensor uncertainty_;

  // Per-node static latent z_s, consumed by BackupNNEvaluator::apply().
  // TODO: populate from the static_latent GPU head once it is wired up; currently zero-filled.
  StaticLatentArray static_latent;

  // BackupNNEvaluator::weight_gen() snapshot taken when this node's NN evaluation was loaded
  // or refreshed. A mismatch against the evaluator's current weight_gen() means this node's
  // weight-dependent state (R, U, static_latent, edge P/AV/AU/z_a, edge e_cached,
  // stats.backup_accumulator) was produced with stale weights and needs refresh. Sentinel -1
  // tags terminal nodes and not-yet-evaluated nodes; the staleness check excludes those.
  int weight_gen = -1;
};

}  // namespace beta0

#include "inline/beta0/NodeStableData.inl"
