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

  int child_N = 0;
  int XC = 0;  // exploration count

  // pi is derived from A via pi = softmax(A)
  // If pi is zero, A will be zero as well. Note however that A == 0 does not necessarily imply
  // pi == 0.
  float A = 0;

  ValueArray child_AV;
  ValueArray child_AU;
  util::Gaussian1D previous_lQW;
  LogitValueArray child_lAUV;
};

}  // namespace beta0
