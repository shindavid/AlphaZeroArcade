#include <util/mit/thread.hpp>

#include <util/Asserts.hpp>
#include <util/LoggingUtil.hpp>
#include <util/mit/scheduler.hpp>

namespace mit {

inline thread::thread() noexcept : impl_(std::make_shared<thread_impl>(this)) {}

inline thread::thread(bool dummy) noexcept
    : impl_(std::make_shared<thread_impl>(this, true, true)) {}

template <typename Function>
thread::thread(Function&& func) : impl_(std::make_shared<thread_impl>(this, true)) {
  auto sched = scheduler::instance();

  auto wrapper = [sched, this, func = std::forward<Function>(func)]() mutable {
    // Note: this can be std::move()'d at any point within this lambda, but this->impl_ will
    // continue to point to the same impl, making this kosher.
    sched->block_until_has_control(this->impl_.get());
    func();
    sched->deactivate_thread(this->impl_.get());
  };

  impl_->std_thread = std::thread(std::move(wrapper));
  sched->pass_control_to(impl_.get());
}

inline thread::thread(thread&& other) noexcept {
  impl_ = other.impl_;  // intentionally do not reset other.impl_
  impl_->owner = this;  // transfer ownership to this thread
}

inline thread& thread::operator=(thread&& other) noexcept {
  impl_ = other.impl_;  // intentionally do not reset other.impl_
  impl_->owner = this;  // transfer ownership to this thread
  return *this;
}

inline thread_impl::thread_impl(thread* t, bool activate, bool skip_registration)
    : owner(t), activated(activate) {
  if (!skip_registration) {
    auto sched = scheduler::instance();
    sched->register_thread(this);
  }
}

inline thread_impl::~thread_impl() {
  auto sched = scheduler::instance();
  sched->unregister_thread(this);
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
  util::release_assert(blocking_cv == cv && !blocking_mutex);
  blocking_cv = nullptr;
}

inline void thread_impl::lift_block(mutex* m) {
  util::release_assert(!blocking_cv && blocking_mutex == m);
  blocking_mutex = nullptr;
}

inline bool thread_impl::viable() const {
  return !blocking_cv && !blocking_mutex && activated;
}

}  // namespace mit
