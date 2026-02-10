#pragma once

#include "alpha0/GameLogFullRecord.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct GameLogFullRecord : public alpha0::GameLogFullRecord<Game> {
  using Base = alpha0::GameLogFullRecord<Game>;
  using ActionValueTensor = Base::ActionValueTensor;
  using WinShareTensor = Game::Types::WinShareTensor;

  WinShareTensor Q;
  WinShareTensor Q_min;  // for each player, the minimum value of Q ever observed for that player
  WinShareTensor Q_max;  // for each player, the maximum value of Q ever observed for that player
  WinShareTensor W;

  // below are valid iff action_values_valid is true
  ActionValueTensor AQ_min;
  ActionValueTensor AQ_max;
  ActionValueTensor AU;
  ActionValueTensor AW;
};

}  // namespace beta0
