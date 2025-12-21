#pragma once

#include "core/BasicTypes.hpp"
#include "core/YieldManager.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
struct ChanceEventHandleRequest {
  using State = Game::State;

  ChanceEventHandleRequest(const YieldNotificationUnit& u, const State& s, action_t ca)
      : notification_unit(u), state(s), chance_action(ca) {}

  const YieldNotificationUnit& notification_unit;
  const State& state;
  action_t chance_action;
};

}  // namespace core
