#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "search/EdgeBase.hpp"

namespace beta0 {

/*
 * An Edge corresponds to an action that can be taken from this node.
 */
template <core::concepts::EvalSpec EvalSpec>
struct Edge : public search::EdgeBase {
  using LogitValueArray = EvalSpec::Game::Types::LogitValueArray;
  using ValueArray = EvalSpec::Game::Types::ValueArray;

  float P;
  float pi;

  int XC = 0;  // exploration count

  float A = 0;  // pi is derived from A via pi = softmax(A)

  // Q of child is modeled as sigmoid(parent_beta + child_lAV + delta)
  //
  // This assumes kNumPlayers <= 2; for more players, delta would need to be a vector.
  float delta = 0.f;

  ValueArray child_AV;
  ValueArray child_AU;
  LogitValueArray child_lAUV;
};

}  // namespace beta0
