#pragma once

#include "core/YieldManager.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
struct ChanceEventHandleRequest {
  using InfoSet = Game::InfoSet;
  using Move = Game::Move;

  ChanceEventHandleRequest(const YieldNotificationUnit& u, const InfoSet& is, Move cm)
      : notification_unit(u), info_set(is), chance_move(cm) {}

  const YieldNotificationUnit& notification_unit;
  const InfoSet& info_set;
  Move chance_move;
};

}  // namespace core
