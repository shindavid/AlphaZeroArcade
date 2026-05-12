#pragma once

#include "beta0/concepts/SpecConcept.hpp"

#include <boost/json.hpp>

namespace beta0 {

/*
 * Per-action statistics captured at the root after a full search when the backup NN is active.
 * Used as training targets for the backup NN. All scalar fields are recorded from the root's
 * active-seat perspective; the trailing 's' on Qs/Ws/AVs/AUs stands for 'seat'.
 *
 *   N[i]:    visit count for action i
 *   Qs[i]:   child Q value (active-seat) for action i
 *   Ws[i]:   child W value (active-seat) for action i
 *   P[i]:    adjusted (post-Dirichlet, post-temperature) policy prior for action i
 *   AVs[i]:  child action-value head estimate (active-seat) for action i, captured from the
 *            edge at parent-eval time
 *   AUs[i]:  child action-value-uncertainty head estimate (active-seat) for action i
 *   Qs_star, Ws_star: the prior-augmented children-average baselines (LoTE/LoTV) at the
 *     root before any backup-NN override, recorded as scalars from the root's active-seat
 *     perspective. These are the context inputs that BackupNet consumes alongside the
 *     accumulator and z_s; we record them so training sees the same baselines the search-time
 *     accumulator was built against.
 *
 * valid is true only when backup_nn_evaluator is ready and it was a full search.
 *
 * The per-action fields N/Qs/Ws/P/AVs/AUs are packed by ChildStatsTarget into a single (A, 6)
 * training-target tensor consumed by the Python ChildEmbeddingHead.
 */
template <beta0::concepts::Spec Spec>
struct BackupSampleData {
  using Game = Spec::Game;
  using TensorEncodings = Spec::TensorEncodings;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ValueArray = Game::Traits::ValueArray;

  bool valid = false;
  PolicyTensor N;
  PolicyTensor Qs;
  PolicyTensor Ws;
  PolicyTensor P;
  PolicyTensor AVs;
  PolicyTensor AUs;
  float Qs_star = 0.0f;
  float Ws_star = 0.0f;

  boost::json::object to_json() const;
};

}  // namespace beta0

#include "inline/beta0/BackupSampleData.inl"
