#pragma once

#include "core/YieldManager.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
struct ChanceEventHandleRequest {
  using State = Game::State;
  using Move = Game::Move;

  ChanceEventHandleRequest(const YieldNotificationUnit& u, const State& s, Move cm)
      : notification_unit(u), state(s), chance_move(cm) {}

  const YieldNotificationUnit& notification_unit;
  const State& state;
  Move chance_move;
};

}  // namespace core
