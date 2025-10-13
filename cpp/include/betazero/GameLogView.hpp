#pragma once

#include "alphazero/GameLogView.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct GameLogView : public alpha0::GameLogView<Game> {
  using Base = alpha0::GameLogView<Game>;
  using ActionValueTensor = Base::ActionValueTensor;
  using WinShareTensor = Game::Types::WinShareTensor;

  ActionValueTensor action_value_uncertainties;
  WinShareTensor Q_posterior;
  WinShareTensor Q_min;  // for each player, the minimum value of Q ever observed for that player
  WinShareTensor Q_max;  // for each player, the maximum value of Q ever observed for that player
  bool action_value_uncertainties_valid;
};

}  // namespace beta0
