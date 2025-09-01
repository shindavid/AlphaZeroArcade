#pragma once

#include "core/BasicTypes.hpp"
#include "core/YieldManager.hpp"

namespace search {

struct SearchRequest {
  SearchRequest(const core::YieldNotificationUnit& u) : notification_unit(u) {}
  SearchRequest() = default;

  core::YieldManager* yield_manager() const { return notification_unit.yield_manager; }
  core::context_id_t context_id() const { return notification_unit.context_id; }
  core::game_slot_index_t game_slot_index() const { return notification_unit.game_slot_index; }

  core::YieldNotificationUnit notification_unit;
};

}  // namespace search
