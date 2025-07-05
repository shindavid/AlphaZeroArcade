#include <util/mit/thread.hpp>

#include <util/Asserts.hpp>
#include <util/mit/scheduler.hpp>

namespace mit {

inline thread::thread() noexcept {
  auto sched = scheduler::instance();
  sched->register_thread(this);
}

inline thread::~thread() {
  auto sched = scheduler::instance();
  sched->unregister_thread(this);
}

template <typename Function>
thread::thread(Function&& func) {
  auto sched = scheduler::instance();
  sched->register_thread(this);

  auto wrapper = [sched, this, func = std::forward<Function>(func)]() mutable {
    sched->pass_control_to(this);
    func();
    sched->deactivate_thread(this);
  };

  thread_ = std::thread(std::move(wrapper));
}

inline thread::thread(thread&& other) noexcept {
  auto sched = scheduler::instance();
  sched->register_thread(this);
  sched->move_thread(&other, this);
  other.move_to(this);
  sched->pass_control_to(this);
}

inline thread& thread::operator=(thread&& other) noexcept {
  if (this != &other) {
    auto sched = scheduler::instance();
    sched->move_thread(&other, this);
    other.move_to(this);
    sched->pass_control_to(this);
  }
  return *this;
}

inline void thread::mark_as_blocked_by(mutex* m) {
  util::debug_assert(!blocking_mutex_);
  blocking_mutex_ = m;
}

inline void thread::lift_block(mutex* m) {
  util::debug_assert(blocking_mutex_ == m);
  blocking_mutex_ = nullptr;
}

inline bool thread::viable() const {
  return !blocking_mutex_ && thread_.joinable();
}

inline void thread::move_to(thread* other) {
  other->thread_ = std::move(thread_);
  other->blocking_mutex_ = blocking_mutex_;

  thread_ = std::thread();
  blocking_mutex_ = nullptr;
}


}  // namespace mit
