#include <util/mit/condition_variable.hpp>

#include <util/mit/scheduler.hpp>

namespace mit {

inline condition_variable::condition_variable() {
  auto sched = scheduler::instance();
  sched->register_condition_variable(this);
}

inline condition_variable::~condition_variable() {
  auto sched = scheduler::instance();
  sched->unregister_condition_variable(this);
}

inline void condition_variable::notify_one() {
  auto sched = scheduler::instance();
  sched->notify_one(this);
}

inline void condition_variable::notify_all() {
  auto sched = scheduler::instance();
  sched->notify_all(this);
}

inline void condition_variable::wait(mit::unique_lock<mit::mutex>& lock) {
  auto sched = scheduler::instance();
  sched->wait_on(this, lock);
}

template <class Predicate>
void condition_variable::wait(mit::unique_lock<mit::mutex>& lock, Predicate pred) {
  auto sched = scheduler::instance();
  sched->wait_on(this, lock, pred);
}

}  // namespace mit
