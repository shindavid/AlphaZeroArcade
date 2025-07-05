#include <util/mit/mutex.hpp>

#include <util/Exception.hpp>
#include <util/mit/scheduler.hpp>

namespace mit {

inline mutex::mutex() {
  auto sched = scheduler::instance();
  sched->register_mutex(this);
}

inline mutex::~mutex() {
  auto sched = scheduler::instance();
  sched->unregister_mutex(this);
}

inline void mutex::lock() {
  auto sched = scheduler::instance();
  if (sched->is_locked(this)) {
    // If the mutex is already locked, we need to yield control to the scheduler
    // to allow other threads to run.
    sched->mark_active_thread_as_blocked_by(this);
    sched->switch_active_thread();
  }

  if (!mutex_.try_lock()) {
    throw util::Exception("Failed to lock mutex");
  }
  sched->mark_as_locked(this);
}

inline void mutex::unlock() {
  mutex_.unlock();
  auto sched = scheduler::instance();
  sched->mark_as_unlocked(this);
}

}  // namespace mit
