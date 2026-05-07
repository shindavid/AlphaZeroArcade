#pragma once

#include "beta0/concepts/SpecConcept.hpp"

#include <boost/json.hpp>

namespace beta0 {

/*
 * Per-action statistics captured at the root after a full search when the backup NN is active.
 * Used as training targets for the backup NN.
 *
 *   N[i]: visit count for action i
 *   Q[i]: child Q value (active-seat perspective) for action i
 *   W[i]: child W value (active-seat perspective) for action i
 *   Qs_star, Ws_star: the prior-augmented children-average baselines (LoTE/LoTV) at the
 *     root before any backup-NN override, recorded as scalars from the root's active-seat
 *     perspective (the trailing 's' stands for 'seat'). These are the context inputs that
 *     BackupNet consumes alongside the accumulator and z_s; we record them so training sees
 *     the same baselines the search-time accumulator was built against.
 *
 * valid is true only when backup_nn_evaluator is ready and it was a full search.
 */
template <beta0::concepts::Spec Spec>
struct BackupSampleData {
  using Game = Spec::Game;
  using TensorEncodings = Spec::TensorEncodings;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ValueArray = Game::Types::ValueArray;

  bool valid = false;
  PolicyTensor N;
  PolicyTensor Q;
  PolicyTensor W;
  float Qs_star = 0.0f;
  float Ws_star = 0.0f;

  // TODO: I think we need P here as well, since the backup NN input includes P. Originally, I
  // thought we would get P from the main NN (with a stop-gradient), but that is a "raw" P, which
  // doesn't include adjustments like Dirichlet noise.

  boost::json::object to_json() const;
};

}  // namespace beta0

#include "inline/beta0/BackupSampleData.inl"
