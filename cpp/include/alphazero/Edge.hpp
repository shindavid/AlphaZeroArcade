#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "search/EdgeBase.hpp"

namespace alpha0 {

/*
 * An Edge corresponds to an action that can be taken from this node.
 */
template <core::concepts::EvalSpec EvalSpec>
struct Edge : public search::EdgeBase {
  using ValueArray = EvalSpec::Game::Types::ValueArray;

  Edge() { child_Q_snapshot.fill(0); }

  int E = 0;  // real or virtual count
  float policy_prior_prob = 0;

  // policy_prior_prob + adjustments (from Dirichlet-noise and softmax-temperature)
  float adjusted_base_prob = 0;

  // child_Q_snapshot is a snapshot of the child value for the current player. It is initialized
  // from the neural network's AV estimate at the time of evaluating the parent, and is updated to
  // the child's actual value during backpropagation.
  //
  // If not for move transpositions, this would always match the child's stats().Q once the child
  // has been expanded. However, because of transpositions, this field can go stale, lagging behind
  // the child's actual value.
  ValueArray child_Q_snapshot;
};

}  // namespace alpha0
