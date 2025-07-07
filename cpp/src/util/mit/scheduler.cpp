#include <util/mit/scheduler.hpp>

#include <util/Asserts.hpp>
#include <util/BoostUtil.hpp>
#include <util/LoggingUtil.hpp>
#include <util/Random.hpp>
#include <util/mit/condition_variable.hpp>
#include <util/mit/thread.hpp>
#include <util/mit/mutex.hpp>

namespace mit {

scheduler* scheduler::instance_ = nullptr;

void scheduler::register_thread(thread_impl* t) {
  util::release_assert(t->id < 0, "Thread already registered ({})", t->id);

  t->id = thread_id_provider_.get_next_id();
  if (t->id >= (int)all_threads_.size()) {
    all_threads_.resize(t->id + 1);
    viable_threads_.resize(t->id + 1);
  }
  all_threads_[t->id] = t;
  viable_threads_[t->id] = t->viable();
  LOG_INFO("Registering thread {} (viable={})", t->id, t->viable());
}

void scheduler::unregister_thread(thread_impl* t) {
  util::release_assert(t->id >= 0, "Thread not registered ({})", t->id);

  all_threads_[t->id] = nullptr;  // Clear the thread from the list
  viable_threads_[t->id] = false;
  thread_id_provider_.recycle(t->id);
  LOG_INFO("Unregistering thread {}", t->id);
}

void scheduler::deactivate_thread(thread_impl* t) {
  LOG_INFO("Deactivating thread {}", t->id);
  t->activated = false;
  viable_threads_[t->id] = false;

  thread_impl* next_thread = get_next_thread();
  {
    std::unique_lock lock(mutex_);
    active_thread_ = next_thread;
  }

  cv_.notify_all();
}

void scheduler::block_until_has_control(thread_impl* t) {
  LOG_INFO("Blocking thread {} until it has control", t->id);
  std::unique_lock lock(mutex_);
  cv_.wait(lock, [&]() { return active_thread_ == t; });
}

void scheduler::pass_control_to(thread_impl* t) {
  LOG_INFO("Passing control from thread {} to thread {}", active_thread_->id, t->id);
  validate_thread_viability(t);

  std::unique_lock lock(mutex_);
  if (active_thread_ == t) return;
  thread_impl* current_thread = active_thread_;
  active_thread_ = t;

  LOG_INFO("Active thread is now {}. Waiting until {} becomes the active thread",
           active_thread_->id, current_thread->id);

  lock.unlock();
  cv_.notify_all();
  lock.lock();
  cv_.wait(lock, [&]() { return active_thread_ == current_thread; });
  LOG_INFO("Wait over. Thread {} has control", active_thread_->id);
}

void scheduler::register_mutex(mutex* m) {
  util::release_assert(m->id_ < 0, "Mutex already registered ({})", m->id_);

  m->id_ = mutex_id_provider_.get_next_id();
  if (m->id_ >= (int)mutex_block_map_.size()) {
    mutex_block_map_.resize(m->id_ + 1, nullptr);
  }
  util::release_assert(mutex_block_map_[m->id_] == nullptr);
}

void scheduler::unregister_mutex(mutex* m) {
  util::release_assert(m->id_ >= 0, "Mutex not registered ({})", m->id_);

  delete mutex_block_map_[m->id_];
  mutex_block_map_[m->id_] = nullptr;
  mutex_id_provider_.recycle(m->id_);
}

void scheduler::mark_active_thread_as_blocked_by(mutex* m) {
  LOG_INFO("Marking active thread {} as blocked by mutex {}", active_thread_->id, m->id_);
  active_thread_->mark_as_blocked_by(m);
  if (!mutex_block_map_[m->id_]) {
    mutex_block_map_[m->id_] = new thread_vec_t();
  }
  mutex_block_map_[m->id_]->push_back(active_thread_);
  viable_threads_[active_thread_->id] = false;
}


void scheduler::mark_as_unlocked(mutex* m) {
  LOG_INFO("Marking mutex {} as unlocked", m->id_);
  thread_vec_t* blocked_threads = mutex_block_map_[m->id_];
  if (!blocked_threads) return;

  for (thread_impl* t : *blocked_threads) {
    t->lift_block(m);
    viable_threads_[t->id] = t->viable();
  }
  blocked_threads->clear();
}

void scheduler::register_condition_variable(condition_variable* cv) {
  util::release_assert(cv->id_ < 0, "Condition variable already registered ({})", cv->id_);

  cv->id_ = cv_id_provider_.get_next_id();
  if (cv->id_ >= (int)cv_block_map_.size()) {
    cv_block_map_.resize(cv->id_ + 1, nullptr);
  }
  util::release_assert(cv_block_map_[cv->id_] == nullptr);
}

void scheduler::unregister_condition_variable(condition_variable* cv) {
  util::release_assert(cv->id_ >= 0, "Condition variable not registered ({})", cv->id_);

  delete cv_block_map_[cv->id_];
  cv_block_map_[cv->id_] = nullptr;
  cv_id_provider_.recycle(cv->id_);
}

void scheduler::notify_one(condition_variable* cv) {
  LOG_INFO("Notifying one thread waiting on condition variable {}", cv->id_);
  thread_predicate_vec_t* blocked_threads = cv_block_map_[cv->id_];
  if (!blocked_threads || blocked_threads->empty()) return;

  int n = blocked_threads->size();

  // notify a random thread:
  auto it = blocked_threads->begin() + util::Random::uniform_sample(0, n);

  thread_impl* t = it->first;
  predicate_t& pred = it->second;
  if (pred()) {
    // If the predicate is satisfied, lift the block and remove the thread from the vector
    t->lift_block(cv);
    viable_threads_[t->id] = t->viable();
    blocked_threads->erase(it);
  }
}

void scheduler::notify_all(condition_variable* cv) {
  LOG_INFO("Notifying all threads waiting on condition variable {}", cv->id_);
  thread_predicate_vec_t* blocked_threads = cv_block_map_[cv->id_];
  if (!blocked_threads || blocked_threads->empty()) return;

  for (auto& pair : *blocked_threads) {
    thread_impl* t = pair.first;
    predicate_t& pred = pair.second;

    if (pred()) {
      t->lift_block(cv);
      viable_threads_[t->id] = t->viable();
    } else {
      tmp_thread_predicate_vec_.push_back(pair);
    }
  }

  std::swap(*blocked_threads, tmp_thread_predicate_vec_);
  tmp_thread_predicate_vec_.clear();
}

void scheduler::wait_on(condition_variable* cv, unique_lock<mutex>& lock) {
  wait_on_helper(cv, lock, []() { return true; });
}

scheduler::scheduler() {
  thread* main_thread = new thread(true);
  register_thread(main_thread->impl_.get());

  util::release_assert(main_thread->id() == 0,
                       "Main thread should have id() 0, got {}", main_thread->id());
  active_thread_ = main_thread->impl_.get();  // Set the main thread as the active thread
}

thread_impl* scheduler::get_next_thread() const {
  int next = boost_util::get_random_set_index(viable_threads_);
  if (next < 0) throw DeadlockException();
  LOG_INFO("Next viable thread index: {}", next);
  return all_threads_[next];
}


void scheduler::validate_thread_viability(thread_impl* t) const {
  int i = t ? t->id : 0;
  int m = all_threads_.size();
  int n = viable_threads_.size();

  util::debug_assert(i >= 0 && i < n, "Thread ID out of range (i={} n={} m={})", i, n, m);
  util::debug_assert(viable_threads_[i], "Thread ID {} is not viable", i);
}

void scheduler::wait_on_helper(condition_variable* cv, unique_lock<mutex>& lock, predicate_t pred) {
  LOG_INFO("Thread {} waiting on condition variable {}", active_thread_->id, cv->id_);
  active_thread_->mark_as_blocked_by(cv);
  if (!cv_block_map_[cv->id_]) {
    cv_block_map_[cv->id_] = new thread_predicate_vec_t();
  }
  cv_block_map_[cv->id_]->push_back(std::make_pair(active_thread_, []() { return true; }));
  viable_threads_[active_thread_->id] = false;
}

}  // namespace mit
