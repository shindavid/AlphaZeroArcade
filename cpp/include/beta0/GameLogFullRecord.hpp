#pragma once

#include "alpha0/GameLogFullRecord.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct GameLogFullRecord : public alpha0::GameLogFullRecord<Game> {
  using Base = alpha0::GameLogFullRecord<Game>;
  using ActionValueTensor = Base::ActionValueTensor;
  using WinShareTensor = Game::Types::WinShareTensor;

  ActionValueTensor AU;  // only valid if AU_valid
  WinShareTensor Q;
  WinShareTensor Q_min;  // for each player, the minimum value of Q ever observed for that player
  WinShareTensor Q_max;  // for each player, the maximum value of Q ever observed for that player
  WinShareTensor W;
  bool AU_valid;
};

}  // namespace beta0
