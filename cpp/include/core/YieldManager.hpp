#pragma once

#include "core/BasicTypes.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <queue>

namespace core {

struct YieldNotificationUnit;

class YieldManager {
 public:
  YieldManager(mit::condition_variable& cv, mit::mutex& mutex, std::queue<SlotContext>& queue,
               int& pending_queue_count)
      : cv_(cv), mutex_(mutex), queue_(queue), pending_queue_count_(pending_queue_count) {}

  void notify(const slot_context_vec_t&);
  void notify(const YieldNotificationUnit&);

 private:
  mit::condition_variable& cv_;
  mit::mutex& mutex_;
  std::queue<SlotContext>& queue_;
  int& pending_queue_count_;
};

struct YieldNotificationUnit {
  YieldNotificationUnit(YieldManager* h, game_slot_index_t g, context_id_t c)
      : yield_manager(h), game_slot_index(g), context_id(c) {}

  YieldNotificationUnit() = default;

  bool valid() const { return yield_manager != nullptr; }
  SlotContext slot_context() const { return SlotContext(game_slot_index, context_id); }

  YieldManager* yield_manager = nullptr;
  game_slot_index_t game_slot_index = -1;
  context_id_t context_id = 0;
};

}  // namespace core

#include "inline/core/YieldManager.inl"
