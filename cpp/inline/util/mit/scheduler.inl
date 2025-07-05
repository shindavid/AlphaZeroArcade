#include <util/mit/scheduler.hpp>

namespace mit {

inline scheduler* scheduler::instance() {
  if (!instance_) {
    instance_ = new scheduler();
  }
  return instance_;
}

inline thread* scheduler::active_thread() { return active_thread_; }

template <class Predicate>
void scheduler::wait_on(condition_variable* cv, unique_lock<mutex>& lock, Predicate pred) {

}

}  // namespace mit
