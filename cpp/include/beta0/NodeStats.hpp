#pragma once

#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct NodeStats {
  using GameTypes = EvalSpec::Game::Types;
  using ValueArray = GameTypes::ValueArray;
  using LogitValueArray = GameTypes::LogitValueArray;

  NodeStats();

  // The logit-normal belief about the value of this node. This is a direct function of Q and W,
  // but we store it here for computational savings.
  LogitValueArray lQW;

  ValueArray Q;
  ValueArray Q_min;  // min Q observed per player
  ValueArray Q_max;  // max Q observed per player
  ValueArray W;      // uncertainty

  int N = 0;    // raw count
  float R = 0;  // relevance-weighted count

  // Child i has a raw value estimate Q_i, and a *beta-adjusted* value estimate
  // sigmoid(beta_parent + lQ_i + delta_i)
  //
  // The beta-adjusted value
  float beta = 0.f;   // shared Q-component across children
};

}  // namespace beta0

#include "inline/beta0/NodeStats.inl"
