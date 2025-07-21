#pragma once

#include "util/Asserts.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

namespace core {

class GameServerClient;

class GameServerBase {
 public:
  enum enqueue_instruction_t : int8_t { kEnqueueNow, kEnqueueLater, kEnqueueNever };

  enum next_result_t : int8_t { kProceed, kHandlePause, kExit };

  virtual ~GameServerBase() = default;

  struct EnqueueRequest {
    enqueue_instruction_t instruction = kEnqueueNow;
    int extra_enqueue_count = 0;  // used when instruction == kEnqueueLater
  };

  struct StepResult {
    EnqueueRequest enqueue_request;
    bool game_ended = false;
    bool drop_slot = false;
  };

  // Helper class used to debug-check that only one thread is ever in a critical section at a time.
  // We could use a mutex, but that would mask the problem, rather than fix it.
  class CriticalSectionCheck {
   public:
    CriticalSectionCheck(std::atomic<bool>& in_critical_section)
        : in_critical_section_(in_critical_section) {
      bool x = in_critical_section_.exchange(true, std::memory_order_acquire);
      RELEASE_ASSERT(!x, "Critical section double-entry detected!");
    }
    ~CriticalSectionCheck() {
      bool x = in_critical_section_.exchange(false, std::memory_order_acquire);
      RELEASE_ASSERT(x, "Critical section double-exit detected!");
    }

   private:
    std::atomic<bool>& in_critical_section_;
  };

  GameServerBase(int num_game_threads) : num_game_threads_(num_game_threads) {}

  void add_client(GameServerClient* client);

  int num_game_threads() const { return num_game_threads_; }

  // Tells the server that it should probably use alternating mode.
  virtual void handle_alternating_mode_recommendation() {}

  virtual void debug_dump() const = 0;

 protected:
  void force_progress();

 private:
  using server_vec_t = std::vector<GameServerBase*>;
  using client_vec_t = std::vector<GameServerClient*>;

  client_vec_t clients_;
  const int num_game_threads_;
};

}  // namespace core
