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
 *
 * valid is true only when backup_nn_evaluator is ready and it was a full search.
 */
template <beta0::concepts::Spec Spec>
struct BackupSampleData {
  using TensorEncodings = Spec::TensorEncodings;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;

  bool valid = false;
  PolicyTensor N;
  PolicyTensor Q;
  PolicyTensor W;

  // TODO: I think we need P here as well, since the backup NN input includes P. Originally, I
  // thought we would get P from the main NN (with a stop-gradient), but that is a "raw" P, which
  // doesn't include adjustments like Dirichlet noise.

  boost::json::object to_json() const;
};

}  // namespace beta0

#include "inline/beta0/BackupSampleData.inl"
