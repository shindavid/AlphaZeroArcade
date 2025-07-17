#pragma once

#include <util/mit/thread.hpp>

#include <util/mit/exceptions.hpp>
#include <util/mit/scheduler.hpp>

#include <util/Asserts.hpp>
#include <util/LoggingUtil.hpp>
#include <util/mit/scheduler.hpp>

namespace mit {

inline thread::thread() : impl_(std::make_shared<thread_impl>(this)) {}

inline thread::thread(bool dummy)
    : impl_(std::make_shared<thread_impl>(this, true, true)) {}

template <typename Function>
thread::thread(Function&& func) : impl_(std::make_shared<thread_impl>(this, true)) {
  auto& sched = scheduler::instance();

  thread_impl* impl = impl_.get();
  auto wrapper = [&sched, impl, func = std::forward<Function>(func)]() mutable {
    // Note: this can be std::move()'d at any point within this lambda, but this->impl_ will
    // continue to point to the same impl, making this kosher.
    try {
      sched.block_until_has_control(impl);
      func();
    } catch (const BugDetectedError& e) {
      sched.handle_bug_detected_error(e);
    } catch (...) {
      sched.handle_exception();
    }
    sched.deactivate_thread(impl);
  };

  impl_->std_thread = std::thread(std::move(wrapper));
  sched.yield_control(impl_.get());
}

inline thread::thread(thread&& other) {
  impl_ = other.impl_;  // intentionally do not reset other.impl_
  impl_->owner = this;  // transfer ownership to this thread
}

inline thread::~thread() { impl_ = nullptr; }

inline thread& thread::operator=(thread&& other) {
  impl_ = other.impl_;  // intentionally do not reset other.impl_
  impl_->owner = this;  // transfer ownership to this thread
  return *this;
}

inline bool thread::joinable() const {
  return impl_->std_thread.joinable();
}

inline void thread::join() {
  if (this != impl_->owner || !impl_->activated) {
    impl_->std_thread.join();
    return;
  }

  scheduler::instance().join_thread(impl_.get());
}

inline thread_impl::thread_impl(thread* t, bool activate, bool skip_registration)
    : owner(t), activated(activate) {
  if (!skip_registration) {
    scheduler::instance().register_thread(this);
  }
}

inline thread_impl::~thread_impl() {
  if (id == 0) return;  // Do not unregister the main thread
  scheduler::instance().unregister_thread(this);
}

inline void thread_impl::mark_as_blocked_by(condition_variable* cv) {
  util::release_assert(!blocking_cv && !blocking_mutex);
  blocking_cv = cv;
}

inline void thread_impl::mark_as_blocked_by(mutex* m) {
  util::release_assert(!blocking_cv && !blocking_mutex);
  blocking_mutex = m;
}

inline void thread_impl::lift_block(condition_variable* cv) {
  blocking_cv = nullptr;
}

inline void thread_impl::lift_block(mutex* m) {
  blocking_mutex = nullptr;
}

inline bool thread_impl::viable() const {
  return !blocking_cv && !blocking_mutex && !joinee && activated;
}

}  // namespace mit
