#include <util/mit/scheduler.hpp>

namespace mit {


inline int id_provider::get_next_id() {
  if (!recycled_ids_.empty()) {
    int id = recycled_ids_.back();
    recycled_ids_.pop_back();
    return id;
  }
  return next_++;
}

inline void id_provider::recycle(int id) {
  recycled_ids_.push_back(id);
}

inline scheduler* scheduler::instance() {
  if (!instance_) {
    instance_ = new scheduler();
  }
  return instance_;
}

inline thread_impl* scheduler::active_thread() { return active_thread_; }

template <class Predicate>
void scheduler::wait_on(condition_variable* cv, unique_lock<mutex>& lock, Predicate pred) {
  if (pred()) {
    // If the predicate is already satisfied, we don't need to wait.
    return;
  }

  wait_on_helper(cv, lock, pred);
}

}  // namespace mit
