#include <util/mit/scheduler.hpp>

#include <util/Asserts.hpp>
#include <util/BoostUtil.hpp>
#include <util/mit/thread.hpp>
#include <util/mit/mutex.hpp>

namespace mit {

scheduler* scheduler::instance_ = nullptr;

void scheduler::register_thread(thread* t) {
  std::unique_lock lock(mutex_);
  util::release_assert(t->thread_id_ < 0, "Thread already registered ({})", t->thread_id_);
  t->thread_id_ = all_threads_.size();
  all_threads_.push_back(t);
  viable_threads_.resize(all_threads_.size(), true);
}

void scheduler::unregister_thread(thread* t) {
  util::release_assert(t->thread_id_ > 0, "Cannot unregister main thread");
  util::release_assert(active_thread_ != t, "Cannot unregister active thread ({})", t->thread_id_);

  validate_thread(t);
  all_threads_.erase(all_threads_.begin() + t->thread_id_);

  size_t n = all_threads_.size();

  // Reset thread ID for the remaining threads
  for (size_t i = t->thread_id_; i < n; ++i) {
    all_threads_[i]->thread_id_ = i;
  }

  // Splice the viable threads bitset to match the new size
  for (size_t i = t->thread_id_; i < n; ++i) {
    viable_threads_[i] = viable_threads_[i + 1];
  }
  viable_threads_.resize(n);
}

void scheduler::pass_control_to(thread* t) {
  std::unique_lock lock(mutex_);
  validate_thread_viability(t);
  if (active_thread_ == t) return;

  thread* current_thread = active_thread_;
  active_thread_ = t;
  cv_.wait(lock, [&]() { return active_thread_ == current_thread; });
}

void scheduler::switch_active_thread() {
  std::unique_lock lock(mutex_);

  int next = boost_util::get_random_set_index(viable_threads_);
  util::release_assert(next >= 0, "No viable threads available for switching");
  active_thread_ = all_threads_[next];

  lock.unlock();
  cv_.notify_all();  // Notify all threads that the active thread has changed
}

void scheduler::move_thread(thread* from, thread* to) {
  std::unique_lock lock(mutex_);
  unregister_thread(from);
  validate_thread(to);

  if (active_thread_ == from) {
    active_thread_ = to;
    lock.unlock();
    cv_.notify_all();  // Notify all threads that the active thread has changed (not needed?)
  }
}

void scheduler::deactivate_thread(thread* t) {
  util::release_assert(t->thread_id_ > 0, "Cannot deactivate main thread");

  validate_thread(t);
  viable_threads_[t->thread_id_] = false;
  switch_active_thread();
}

void scheduler::register_mutex(mutex* m) {
  util::release_assert(m->mutex_id_ < 0, "Mutex already registered ({})", m->mutex_id_);
  m->mutex_id_ = all_mutexes_.size();
  all_mutexes_.push_back(m);
  locked_mutexes_.resize(all_mutexes_.size(), false);
}

void scheduler::unregister_mutex(mutex* m) {
  util::release_assert(all_mutexes_.size() == locked_mutexes_.size(),
                      "all_mutexes_ and locked_mutexes_ not in sync ({} != {})",
                      all_mutexes_.size(), locked_mutexes_.size());

  validate_mutex(m);
  all_mutexes_.erase(all_mutexes_.begin() + m->mutex_id_);

  size_t n = all_mutexes_.size();

  // Reset mutex ID for the remaining mutexes
  for (size_t i = m->mutex_id_; i < n; ++i) {
    all_mutexes_[i]->mutex_id_ = i;
  }

  // Splice the locked mutexes bitset to match the new size
  for (size_t i = m->mutex_id_; i < n; ++i) {
    locked_mutexes_[i] = locked_mutexes_[i + 1];
  }
  locked_mutexes_.resize(n);
}

void scheduler::mark_active_thread_as_blocked_by(mutex* m) {
  std::unique_lock lock(mutex_);
  validate_thread(active_thread_);

  active_thread_->mark_as_blocked_by(m);
  mutex_block_map_[m].push_back(active_thread_);
  viable_threads_[active_thread_->thread_id_] = false;
}

bool scheduler::is_locked(mutex* m) const {
  validate_mutex(m);
  return locked_mutexes_[m->mutex_id_];
}

void scheduler::mark_as_locked(mutex* m) {
  validate_mutex(m);
  locked_mutexes_[m->mutex_id_] = true;
}

void scheduler::mark_as_unlocked(mutex* m) {
  validate_mutex(m);
  locked_mutexes_[m->mutex_id_] = false;

  auto it = mutex_block_map_.find(m);
  if (it == mutex_block_map_.end()) return;

  thread_vec_t& blocked_threads = it->second;
  for (thread* t : blocked_threads) {
    t->lift_block(m);
    viable_threads_[t->thread_id_] = t->viable();
  }
  mutex_block_map_.erase(it);
}

void scheduler::register_condition_variable(condition_variable* cv) {
  throw util::Exception("TODO");
}

void scheduler::unregister_condition_variable(condition_variable* cv) {
    throw util::Exception("TODO");
}

void scheduler::notify_one(condition_variable* cv) {
    throw util::Exception("TODO");
}

void scheduler::notify_all(condition_variable* cv) {
    throw util::Exception("TODO");
}

void scheduler::wait_on(condition_variable* cv, unique_lock<mutex>& lock) {
    throw util::Exception("TODO");
}

scheduler::scheduler() {
  thread* main_thread = new thread(true);
  register_thread(main_thread);
  util::release_assert(main_thread->thread_id_ == 0,
                       "Main thread should have thread_id_ 0, got {}", main_thread->thread_id_);
  active_thread_ = main_thread;  // Set the main thread as the active thread
}

void scheduler::validate_thread(thread* t) const {
  int i = t->thread_id_;
  util::release_assert(i >= 0 && i < (int)all_threads_.size(),
                       "Thread ID out of range ({}), all_threads_ size: {}", i,
                       all_threads_.size());
  util::release_assert(all_threads_[i] == t,
                       "Thread at index {} does not match expected thread pointer", i);
}

void scheduler::validate_thread_viability(thread* t) const {
  int i = t ? t->thread_id_ : 0;
  int n = viable_threads_.size();

  util::release_assert(i >= 0 && i < n, "Thread ID out of range (i={} n={})", i, n);
  util::release_assert(viable_threads_[i], "Thread ID {} is not viable", i);
}

void scheduler::validate_mutex(mutex* m) const {
  int i = m->mutex_id_;
  util::release_assert(i >= 0 && i < (int)all_mutexes_.size(),
                       "Mutex ID out of range ({}), all_mutexes_ size: {}", i, all_mutexes_.size());
  util::release_assert(all_mutexes_[i] == m,
                       "Mutex at index {} does not match expected mutex pointer", i);
}

}  // namespace mit
