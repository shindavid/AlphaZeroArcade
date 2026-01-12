#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "search/EdgeBase.hpp"
#include "util/Gaussian1D.hpp"

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
  int RC = 0;  // refresh count

  // A is a Bradley-Terry rating.
  //
  // pi is derived from A via pi = softmax(A)
  //
  // A = 0 means negative infinity; otherwise score should be < 0. For consistency, A is calibrated
  // so that the maximum A among sibling edges is always -1.
  float A = 0;

  util::Gaussian1D child_lQW;  // last belief about child value for active seat
  ValueArray child_AV;
  ValueArray child_AU;
  LogitValueArray child_lAUV;
};

}  // namespace beta0
