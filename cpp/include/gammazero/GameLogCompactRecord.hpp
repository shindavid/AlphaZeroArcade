#pragma once

#include "alphazero/GameLogCompactRecord.hpp"
#include "core/concepts/GameConcept.hpp"

namespace gamma0 {

template <core::concepts::Game Game>
struct GameLogCompactRecord : public alpha0::GameLogCompactRecord<Game> {
  using WinShareTensor = Game::Types::WinShareTensor;
  WinShareTensor Q_posterior;
  WinShareTensor Q_min;  // for each player, the minimum value of Q ever observed for that player
  WinShareTensor Q_max;  // for each player, the maximum value of Q ever observed for that player
  WinShareTensor W_max;  // for each player, the maximum uncertainty ever observed for that player
};

}  // namespace gamma0
