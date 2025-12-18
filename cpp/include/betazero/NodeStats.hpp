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

  ValueArray minus_shocks;  // per-player negative shocks, in logit space
  ValueArray plus_shocks;   // per-player positive shocks, in logit space

  ValueArray Q;
  ValueArray Q_min;    // min Q observed per player
  ValueArray Q_max;    // for each player, the max Q ever observed for that player
  ValueArray W;        // uncertainty
  ValueArray W_max;    // for each player, the max W ever observed for that player
};

}  // namespace beta0

#include "inline/betazero/NodeStats.inl"
