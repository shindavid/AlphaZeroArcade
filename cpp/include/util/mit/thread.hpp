#pragma once

#include <thread>
#include <set>

#ifndef MIT_TEST_MODE
static_assert(false, "This file is not intended to be #include'd directly.");
#endif  // MIT_TEST_MODE

namespace mit {

class mutex;

// Drop-in replacement for std::thread that can be used in unit tests.
//
// Not all std::thread functionality is implemented in this class. Only those methods that are
// used in the AlphaZeroArcade codebase are provided. In the future, we can extend this class to
// include more functionality as needed.
class thread {
 public:
  friend class scheduler;

  thread() noexcept;
  template <typename Function> explicit thread(Function&& func);
  ~thread();

  thread(thread&& other) noexcept;
  thread& operator=(thread&& other) noexcept;

  thread(const thread&) = delete;
  thread& operator=(const thread&) = delete;

  bool joinable() const noexcept { return thread_.joinable(); }
  void join() { return thread_.join(); }

 private:
  void mark_as_blocked_by(mutex* m);
  void lift_block(mutex* m);
  bool viable() const;
  void move_to(thread* other);
  using mutex_set_t = std::set<mutex*>;

  std::thread thread_;  // not set for main thread
  mutex_set_t blocked_by_mutexes_;  // Mutexes that this thread is currently blocked on
  int thread_id_ = -1;              // 0 for main thread
};

}  // namespace mit

#include <inline/util/mit/thread.inl>
