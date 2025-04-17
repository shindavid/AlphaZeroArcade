#pragma once

#include <core/BasicTypes.hpp>

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace core {

class HibernationManager {
 public:
  using game_slot_vec_t = std::vector<game_slot_index_t>;
  using func_t = std::function<void(game_slot_index_t)>;

  // Launches a loop that awaits notifications for game-slots. Each time game-slot g is notified, it
  // calls the function f(g).
  void run(func_t);

  // Shuts down the loop.
  void shut_down();

  void notify(game_slot_index_t slot_id);

 private:
  void loop(func_t f);

  std::condition_variable cv_;
  std::mutex mutex_;
  std::thread thread_;
  game_slot_vec_t ready_slots_;
  bool shutting_down_ = false;
};

}  // namespace core

#include <inline/core/HibernationManager.inl>
