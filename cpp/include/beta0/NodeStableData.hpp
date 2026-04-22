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
  using GameOutcome = Game::Types::GameOutcome;
  using ValueArray = Game::Types::ValueArray;
  using AccumulatorArray = SpecTraits<Spec>::AccumulatorArray;

  // WinShareTensor = FTensor<Sizes<kNumPlayers>> — per-player uncertainties
  using WinShareTensor = TensorEncodings::WinShareTensor;

  NodeStableData(const State&, int n_valid_moves, core::seat_index_t);
  NodeStableData(const State&, const GameOutcome&);

  // Returns the prior uncertainty as a ValueArray (U from the neural network).
  ValueArray U() const { return Eigen::Map<const ValueArray>(uncertainty_.data()); }

  WinShareTensor uncertainty_;

  // Precomputed static portion of the backup-NN accumulator: W_AS @ [z, V, U, P].
  // Populated by load_evaluations() from the "backup_accu_static" NN head.
  AccumulatorArray backup_accu_static;
};

}  // namespace beta0

#include "inline/beta0/NodeStableData.inl"
