#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "search/EdgeBase.hpp"
#include "util/Gaussian1D.hpp"

namespace beta0 {

/*
 * An Edge corresponds to an action that can be taken from this node.
 */
template <core::concepts::EvalSpec EvalSpec>
struct Edge : public search::EdgeBase<EvalSpec> {
  using LogitValueArray = EvalSpec::Game::Types::LogitValueArray;
  using ValueArray = EvalSpec::Game::Types::ValueArray;

  float P_raw;
  float P_adjusted;

  int E = 0;
  int child_N = 0;

  ValueArray child_AV;
  ValueArray child_AU;
  util::Gaussian1D lVUs;
};

}  // namespace beta0
