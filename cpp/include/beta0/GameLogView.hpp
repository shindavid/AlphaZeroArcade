#pragma once

#include "alpha0/GameLogView.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct GameLogView : public alpha0::GameLogView<EvalSpec> {
  using Base = alpha0::GameLogView<EvalSpec>;
  using Game = EvalSpec::Game;
  using ActionValueTensor = Base::ActionValueTensor;
  using WinShareTensor = EvalSpec::TensorEncodings::WinShareTensor;

  WinShareTensor Q;
  WinShareTensor Q_min;  // for each player, the minimum value of Q ever observed for that player
  WinShareTensor Q_max;  // for each player, the maximum value of Q ever observed for that player
  WinShareTensor W;

  // below are valid iff action_values_valid is true
  ActionValueTensor AQ_min;
  ActionValueTensor AQ_max;
  ActionValueTensor AU;
};

}  // namespace beta0
