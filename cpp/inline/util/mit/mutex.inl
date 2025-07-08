#include <util/mit/mutex.hpp>

#include <util/Asserts.hpp>
#include <util/LoggingUtil.hpp>
#include <util/mit/scheduler.hpp>

namespace mit {

inline mutex::mutex() {
  scheduler::instance()->register_mutex(this);
}

inline mutex::~mutex() {
  scheduler::instance()->unregister_mutex(this);
}

inline void mutex::lock() {
  auto sched = scheduler::instance();

  if (locked_) {
    // If the mutex is already locked, we need to yield control to the scheduler
    // to allow other threads to run.
    sched->mark_active_thread_as_blocked_by(this);
    sched->yield_control();
  }

  util::release_assert(!locked_, "Mutex is already locked");
  locked_ = true;
}

inline void mutex::unlock() {
  locked_ = false;
  auto sched = scheduler::instance();
  sched->mark_as_unlocked(this);
  sched->yield_control();
}

}  // namespace mit
