#pragma once

#include <core/BasicTypes.hpp>

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace core {

class YieldManager {
 public:
  using func_t = std::function<void(const slot_context_vec_t&)>;

  ~YieldManager();

  // Launches a loop that awaits notifications for game-slots. Each time game-slot g is notified, it
  // calls the function f(g).
  void run(func_t);

  // Shuts down the loop.
  void shut_down();

  void notify(const core::slot_context_vec_t&);
  void notify(const SlotContext&);

 private:
  void loop(func_t f);

  std::condition_variable cv_;
  std::mutex mutex_;
  std::thread thread_;
  slot_context_vec_t ready_items_;
  bool shutting_down_ = false;
};

struct YieldNotificationUnit {
  YieldNotificationUnit(core::YieldManager* h, core::game_slot_index_t g,
                              core::context_id_t c)
      : yield_manager(h), game_slot_index(g), context_id(c) {}

  YieldNotificationUnit() = default;

  bool valid() const { return yield_manager != nullptr; }
  SlotContext slot_context() const { return SlotContext(game_slot_index, context_id); }

  core::YieldManager* yield_manager = nullptr;
  core::game_slot_index_t game_slot_index = -1;
  core::context_id_t context_id = 0;
};

}  // namespace core

#include <inline/core/YieldManager.inl>
