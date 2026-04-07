#pragma once

#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct NodeStats {
  using GameTraits = EvalSpec::Game::Types;
  using ValueArray = GameTraits::ValueArray;
  using LogitValueArray = GameTraits::LogitValueArray;

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
  bool move_forced = false;
};

}  // namespace beta0

#include "inline/beta0/NodeStats.inl"
