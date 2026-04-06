#pragma once

#include "alpha0/GameLogCompactRecord.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct GameLogCompactRecord : public alpha0::GameLogCompactRecord<EvalSpec> {
  using Game = EvalSpec::Game;
  using WinShareTensor = EvalSpec::TensorEncodings::WinShareTensor;

  WinShareTensor Q;
  WinShareTensor Q_min;  // for each player, the minimum value of Q ever observed for that player
  WinShareTensor Q_max;  // for each player, the maximum value of Q ever observed for that player
  WinShareTensor W;
};

}  // namespace beta0
