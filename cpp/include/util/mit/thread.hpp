#pragma once

#include <memory>
#include <thread>

#ifndef MIT_TEST_MODE
static_assert(false, "This file is not intended to be #include'd directly.");
#endif  // MIT_TEST_MODE

namespace mit {

class condition_variable;
class mutex;
class thread;

// The implementation details of the thread class are encapsulated in this struct, and mit::thread
// merely holds a shared pointer to it. This allows us to implement move semantics in a simple
// way.
struct thread_impl {
  thread_impl(thread* t, bool activate=false, bool skip_registration=false);
  ~thread_impl();

  void mark_as_blocked_by(condition_variable* cv);
  void mark_as_blocked_by(mutex* m);
  void lift_block(condition_variable* cv);
  void lift_block(mutex* m);
  bool viable() const;

  // owner is the thread that owns this impl. If the parent thread is std::move()'d, the owner
  // will be set to the new thread instance. The defunct thread will continue to have its impl_
  // pointer set to this impl, but it will no longer be the owner. Maintaining this defunct
  // thread pointer makes bookkeeping easier.
  thread* owner = nullptr;

  std::thread std_thread;  // not set for main thread
  condition_variable* blocking_cv = nullptr;
  mutex* blocking_mutex = nullptr;
  int id = -1;  // set by scheduler, 0 for main thread
  bool activated = false;
};
using thread_impl_ptr = std::shared_ptr<thread_impl>;

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
  ~thread() { impl_ = nullptr; }

  thread(thread&& other) noexcept;
  thread& operator=(thread&& other) noexcept;

  thread(const thread&) = delete;
  thread& operator=(const thread&) = delete;

  bool joinable() const noexcept { return this == owner() && impl_->std_thread.joinable(); }
  void join();

 private:
  // Special constructor for the main thread.
  thread(bool dummy) noexcept;

  thread* owner() const { return impl_->owner; }

  void mark_as_blocked_by(condition_variable* cv) { impl_->mark_as_blocked_by(cv); }
  void mark_as_blocked_by(mutex* m) { impl_->mark_as_blocked_by(m); }
  void lift_block(condition_variable* cv) { impl_->lift_block(cv); }
  void lift_block(mutex* m) { impl_->lift_block(m); }
  bool viable() const { return impl_->viable(); }
  int id() const { return impl_->id; }

  thread_impl_ptr impl_;
};

}  // namespace mit

#include <inline/util/mit/thread.inl>
