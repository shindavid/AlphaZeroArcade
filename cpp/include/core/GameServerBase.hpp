#pragma once

#include <util/Asserts.hpp>

#include <atomic>
#include <cstdint>
#include <vector>

namespace core {

class GameServerClient;

class GameServerBase {
 public:
  enum enqueue_instruction_t : int8_t { kEnqueueNow, kEnqueueLater, kEnqueueNever };

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
      util::release_assert(!x, "Critical section double-entry detected!");
    }
    ~CriticalSectionCheck() {
      bool x = in_critical_section_.exchange(false, std::memory_order_acquire);
      util::release_assert(x, "Critical section double-exit detected!");
    }

   private:
    std::atomic<bool>& in_critical_section_;
  };

  GameServerBase(int num_game_threads) : num_game_threads_(num_game_threads) {
    game_servers_.push_back(this);
  }

  static void add_client(GameServerClient* client);

  int num_game_threads() const {
    return num_game_threads_;
  }

  virtual void debug_dump() const = 0;

 protected:
  void force_progress();

 private:
  using server_vec_t = std::vector<GameServerBase*>;
  using client_vec_t = std::vector<GameServerClient*>;

  static server_vec_t game_servers_;
  client_vec_t clients_;
  const int num_game_threads_;
};

}  // namespace core
