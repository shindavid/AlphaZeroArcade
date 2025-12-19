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
  int RC = 0;  // refresh count

  ValueArray child_AV;
  ValueArray child_AU;
  LogitValueArray child_lAUV;
};

}  // namespace beta0
