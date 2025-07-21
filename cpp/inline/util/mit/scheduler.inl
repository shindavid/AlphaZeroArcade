#pragma once

#include <util/mit/scheduler.hpp>

#include <util/Asserts.hpp>
#include <util/BoostUtil.hpp>
#include <util/LoggingUtil.hpp>
#include <util/Random.hpp>
#include <util/mit/condition_variable.hpp>
#include <util/mit/exceptions.hpp>
#include <util/mit/logging.hpp>
#include <util/mit/thread.hpp>
#include <util/mit/mutex.hpp>

#include <ctime>
#include <exception>
#include <format>
#include <sstream>

namespace mit {

inline scheduler::~scheduler() noexcept(false) {
  // If any threads are still joinable, that's a potential bug.
  for (const auto& thread : all_threads_) {
    if (thread && thread->std_thread.joinable()) {
      throw BugDetectedError("Program exiting with joinable threads");
    }
  }
}

inline scheduler& scheduler::instance() {
  static scheduler instance;
  return instance;
}

inline void scheduler::reset() {
  thread_id_provider_.clear();
  all_threads_.clear();
  viable_threads_.clear();
  active_thread_ = nullptr;

  mutex_id_provider_.clear();
  for (auto* blocked_threads : mutex_block_map_) {
    delete blocked_threads;
  }
  mutex_block_map_.clear();

  cv_id_provider_.clear();
  for (auto* blocked_threads : cv_block_map_) {
    delete blocked_threads;
  }
  cv_block_map_.clear();

  tmp_thread_predicate_vec_.clear();

  caught_exception_ = nullptr;
  uncaught_throw_count_ = 0;
  bug_catching_enabled_ = false;
  mid_orderly_shutdown_ = false;

  init_main_thread();
}

inline void scheduler::register_thread(thread_impl* t) {
  RELEASE_ASSERT(t->id < 0, "Thread already registered ({})", t->id);

  t->id = thread_id_provider_.get_next_id();
  if (t->id >= (int)all_threads_.size()) {
    all_threads_.resize(t->id + 1);
    viable_threads_.resize(t->id + 1);
  }
  all_threads_[t->id] = t;
  viable_threads_[t->id] = t->viable();
  MIT_LOG("Registered thread {} (viable={})", t->id, t->viable());
  debug_dump_state();
}

inline void scheduler::unregister_thread(thread_impl* t) {
  RELEASE_ASSERT(t->id >= 0, "Thread not registered ({})", t->id);

  all_threads_[t->id] = nullptr;  // Clear the thread from the list
  viable_threads_[t->id] = false;
  thread_id_provider_.recycle(t->id);
  MIT_LOG("Unregistered thread {}", t->id);
  debug_dump_state();
}

inline thread_impl* scheduler::active_thread() { return active_thread_; }

inline void scheduler::join_thread(thread_impl* t) {
  thread_impl* current_thread = active_thread_;
  MIT_LOG("Joining thread {} (current thread={})", t->id, current_thread->id);

  current_thread->joinee = t;
  t->joiner = current_thread;
  viable_threads_[current_thread->id] = false;
  try {
    pass_control_to_once_viable(t);
  } catch (const BugDetectedError& e) {
    MIT_LOG("Caught BugDetectedError while joining thread {}: {}", t->id, e.what());
    handle_bug_detected_error(e);
    pass_control_to_once_viable(t);
  }

  if (t->std_thread.joinable()) {
    MIT_LOG("Thread {} is joinable, joining...", t->id);
    t->std_thread.join();
  } else {
    MIT_LOG("Thread {} is not joinable, skipping join", t->id);
  }

  current_thread->joinee = nullptr;
  viable_threads_[current_thread->id] = current_thread->viable();

  MIT_LOG("Thread {} joined! Passing back to thread {}", t->id, current_thread->id);
  pass_control_to(current_thread);
}

inline void scheduler::yield_control(thread_impl* t) {
  thread_impl* current_thread = active_thread_;
  if (!t) {
    t = get_next_thread();
    if (!t) {
      throw_bug_detected_error(__func__);
    }
  }
  if (t == current_thread) {
    return;
  }
  MIT_LOG("Yielding control from thread {} to thread {}", current_thread->id, t->id);
  pass_control_to(t);
  block_until_has_control(current_thread);
}

inline void scheduler::deactivate_thread(thread_impl* t) {
  std::unique_lock lock(mutex_);
  MIT_LOG("Deactivating thread {}", t->id);
  t->activated = false;
  viable_threads_[t->id] = false;

  thread_impl* joiner = t->joiner;
  thread_impl* next_thread = nullptr;
  if (joiner) {
    RELEASE_ASSERT(joiner->joinee == t);
    MIT_LOG("Reactivating thread {}, which was joining {}", joiner->id, t->id);
    joiner->joinee = nullptr;
    t->joiner = nullptr;
    viable_threads_[joiner->id] = joiner->viable();
    if (joiner->viable()) {
      next_thread = joiner;
    }
  }

  if (!next_thread) {
    next_thread = get_next_thread();
    if (!next_thread) {
      try {
        throw_bug_detected_error(__func__);
      } catch (const BugDetectedError& e) {
        handle_bug_detected_error(e);
      }
      next_thread = get_next_thread();
      RELEASE_ASSERT(next_thread, "No viable threads available after bug handling");
    }
  }
  active_thread_ = next_thread;

  MIT_LOG("Deactivation of thread {} complete", t->id);
  debug_dump_state();

  lock.unlock();
  cv_.notify_all();
}

inline void scheduler::block_until_has_control(thread_impl* t) {
  MIT_LOG("Blocking until thread {} has control", t->id);
  std::unique_lock lock(mutex_);
  cv_.wait(lock, [&]() { return active_thread_ == t; });
  MIT_LOG("Thread {} now has control", t->id);
  debug_dump_state();
}

inline void scheduler::register_mutex(mutex* m) {
  RELEASE_ASSERT(m->id_ < 0, "Mutex already registered ({})", m->id_);

  m->id_ = mutex_id_provider_.get_next_id();
  if (m->id_ >= (int)mutex_block_map_.size()) {
    mutex_block_map_.resize(m->id_ + 1, nullptr);
  }
  RELEASE_ASSERT(mutex_block_map_[m->id_] == nullptr);
}

inline void scheduler::unregister_mutex(mutex* m) {
  RELEASE_ASSERT(m->id_ >= 0, "Mutex not registered ({})", m->id_);

  delete mutex_block_map_[m->id_];
  mutex_block_map_[m->id_] = nullptr;
  mutex_id_provider_.recycle(m->id_);
}

inline void scheduler::lock_mutex(mutex* m) {
  if (mid_orderly_shutdown_) {
    throw OrderlyShutdownException();
  }

  while (m->locked_) {
    // If the mutex is already locked, we need to yield control to the scheduler
    // to allow other threads to run.
    mark_active_thread_as_blocked_by(m);
    yield_control();
  }

  MIT_LOG("Locking mutex {}", m->id_);
  RELEASE_ASSERT(!m->locked_, "Mutex is already locked");
  m->locked_ = true;
  yield_control();
}

inline void scheduler::unlock_mutex(mutex* m) {
  MIT_LOG("Unlocking mutex {}", m->id_);
  m->locked_ = false;
  mark_as_unlocked(m);
  yield_control();
}

inline void scheduler::mark_active_thread_as_blocked_by(mutex* m) {
  MIT_LOG("Marking active thread {} as blocked by mutex {}", active_thread_->id, m->id_);
  active_thread_->mark_as_blocked_by(m);
  if (!mutex_block_map_[m->id_]) {
    mutex_block_map_[m->id_] = new thread_vec_t();
  }
  mutex_block_map_[m->id_]->push_back(active_thread_);
  viable_threads_[active_thread_->id] = false;
  debug_dump_state();
}


inline void scheduler::mark_as_unlocked(mutex* m) {
  thread_vec_t* blocked_threads = mutex_block_map_[m->id_];
  if (!blocked_threads || blocked_threads->empty()) return;

  for (thread_impl* t : *blocked_threads) {
    t->lift_block(m);
    viable_threads_[t->id] = t->viable();
    MIT_LOG("Thread {} no longer blocked by mutex {}", t->id, m->id_);
  }
  blocked_threads->clear();
  debug_dump_state();
}

inline void scheduler::register_condition_variable(condition_variable* cv) {
  RELEASE_ASSERT(cv->id_ < 0, "Condition variable already registered ({})", cv->id_);

  cv->id_ = cv_id_provider_.get_next_id();
  if (cv->id_ >= (int)cv_block_map_.size()) {
    cv_block_map_.resize(cv->id_ + 1, nullptr);
  }
  RELEASE_ASSERT(cv_block_map_[cv->id_] == nullptr);
}

inline void scheduler::unregister_condition_variable(condition_variable* cv) {
  RELEASE_ASSERT(cv->id_ >= 0, "Condition variable not registered ({})", cv->id_);

  delete cv_block_map_[cv->id_];
  cv_block_map_[cv->id_] = nullptr;
  cv_id_provider_.recycle(cv->id_);
}

inline void scheduler::notify_one(condition_variable* cv) {
  thread_predicate_vec_t* blocked_threads = cv_block_map_[cv->id_];
  if (!blocked_threads || blocked_threads->empty()) return;

  int n = blocked_threads->size();

  // notify a random thread:
  auto it = blocked_threads->begin() + util::Random::uniform_sample(prng_, 0, n);

  thread_impl* t = it->first;
  predicate_t& pred = it->second;
  if (pred()) {
    MIT_LOG("Notifying thread {}, which was blocked on condition variable {}", t->id, cv->id_);
    // If the predicate is satisfied, lift the block and remove the thread from the vector
    t->lift_block(cv);
    viable_threads_[t->id] = t->viable();
    blocked_threads->erase(it);
    debug_dump_state();
  }
}

inline void scheduler::notify_all(condition_variable* cv) {
  thread_predicate_vec_t* blocked_threads = cv_block_map_[cv->id_];
  if (!blocked_threads || blocked_threads->empty()) return;

  for (auto& pair : *blocked_threads) {
    thread_impl* t = pair.first;
    predicate_t& pred = pair.second;

    if (pred()) {
      MIT_LOG("Notifying thread {}, which was blocked on condition variable {}", t->id, cv->id_);
      t->lift_block(cv);
      viable_threads_[t->id] = t->viable();
    } else {
      tmp_thread_predicate_vec_.push_back(pair);
    }
  }

  std::swap(*blocked_threads, tmp_thread_predicate_vec_);
  tmp_thread_predicate_vec_.clear();
  debug_dump_state();
}

inline void scheduler::wait_on(condition_variable* cv, unique_lock<mutex>& lock) {
  wait_on_helper(cv, lock, []() { return true; });
}

template <class Predicate>
void scheduler::wait_on(condition_variable* cv, unique_lock<mutex>& lock, Predicate pred) {
  if (pred()) {
    // If the predicate is already satisfied, we don't need to wait.
    MIT_LOG("Predicate already satisfied, not waiting on condition variable {}", cv->id_);
    return;
  }

  wait_on_helper(cv, lock, pred);
}

inline void scheduler::enable_bug_catching_mode() {
  bug_catching_enabled_ = true;
}

inline void scheduler::disable_bug_catching_mode() {
  bug_catching_enabled_ = false;
  if (!caught_exception_) return;
  if (uncaught_throw_count_ > 0) return;

  MIT_LOG("{}(), caught exception", __func__);
  std::exception_ptr caught_exception_copy = std::move(caught_exception_);
  reset();
  std::rethrow_exception(caught_exception_copy);
}

inline void scheduler::handle_bug_detected_error(const BugDetectedError& e) {
  if (!bug_catching_enabled_) throw e;  // Re-throw the error if bug catching is not enabled

  uncaught_throw_count_--;
  if (!caught_exception_) {
    caught_exception_ = std::make_exception_ptr(e);
  }
}

inline void scheduler::throw_bug_detected_error(const char* func) {
  MIT_LOG("Throwing BugDetectedError from {}", func);
  if (bug_catching_enabled_ && !mid_orderly_shutdown_) {
    commence_orderly_shutdown();
  }

  if (active_thread_->id == 0) {
    // If the main thread is the one that detected the bug, we need to pre-emptively join the
    // pending threads to avoid a std::terminate() due to the destruction of a joinable std::thread.

    // Set caught_exception_ to ensure proper downstream exception handling
    try {
      throw BugDetectedError();
    } catch (const BugDetectedError& e) {
      if (!caught_exception_) {
        caught_exception_ = std::make_exception_ptr(e);
      }
    }

    MIT_LOG("Yielding to all threads before throwing BugDetectedError");
    while (viable_threads_.any()) {
      size_t n = viable_threads_.find_first();
      if (n == 0) {
        n = viable_threads_.find_next(n);
        if (n == boost::dynamic_bitset<>::npos) {
          break;  // No more viable threads
        }
      }
      yield_control(all_threads_[n]);
    }
    for (thread_impl* t : all_threads_) {
      if (t && t->std_thread.joinable()) {
        MIT_LOG("Joining thread {}", t->id);
        t->std_thread.join();
      }
    }
    MIT_LOG("Done joining all threads");
  }
  uncaught_throw_count_++;
  throw BugDetectedError();
}

inline void scheduler::commence_orderly_shutdown() {
  MIT_LOG("Commencing orderly shutdown...");
  mid_orderly_shutdown_ = true;

  // Clear all mutex/cv blocks
  for (thread_impl* t : all_threads_) {
    if (t) {
      if (t->blocking_mutex) {
        t->blocking_mutex->locked_ = false;
        t->blocking_mutex = nullptr;
      }
      t->blocking_cv = nullptr;
      viable_threads_[t->id] = t->viable();
    }
  }

  for (thread_vec_t* blocked_threads : mutex_block_map_) {
    if (blocked_threads) {
      blocked_threads->clear();  // Clear all mutex blocks
    }
  }

  for (thread_predicate_vec_t* blocked_threads : cv_block_map_) {
    if (blocked_threads) {
      blocked_threads->clear();  // Clear all cv blocks
    }
  }

  debug_dump_state();
  cv_.notify_all();
}

inline void scheduler::handle_exception() {
  if (mid_orderly_shutdown_) return;  // Ignore subsequent exceptions during shutdown
  throw;
}

inline void scheduler::debug_dump_state() const {
  if (mit::kEnableDebugLogging) {
    dump_state();
  }
}

inline void scheduler::dump_state() const {
  LOG_INFO("Scheduler state:");

  for (size_t i = 0; i < all_threads_.size(); ++i) {
    thread_impl* t = all_threads_[i];
    if (t) {
      std::stringstream ss;
      bool active = (t == active_thread_);
      ss << std::format("{} Thread {}: viable={}", active ? '*' : ' ', i, viable_threads_[i]);
      if (t->blocking_mutex) {
        ss << " blocked_by=m" << t->blocking_mutex->id_;
      }
      if (t->blocking_cv) {
        ss << " blocked_by=cv" << t->blocking_cv->id_;
      }
      if (t->joiner) {
        ss << " joined_by=" << t->joiner->id;
      }
      if (t->joinee) {
        ss << " joining=" << t->joinee->id;
      }
      LOG_INFO("{}", ss.str());
    }
  }
}

inline scheduler::scheduler() {
  seed(std::time(nullptr));
  init_main_thread();
}

inline void scheduler::init_main_thread() {
  delete main_thread_;
  main_thread_ = new thread(true);
  register_thread(main_thread_->impl_.get());

  RELEASE_ASSERT(main_thread_->id() == 0,
                       "Main thread should have id() 0, got {}", main_thread_->id());
  active_thread_ = main_thread_->impl_.get();  // Set the main thread as the active thread
}

inline void scheduler::pass_control_to(thread_impl* t) {
  std::unique_lock lock(mutex_);

  if (active_thread_ == t) return;

  MIT_LOG("Passing control from thread {} to thread {}", active_thread_->id, t->id);
  if (!t->viable()) {
    throw_bug_detected_error(__func__);
  }
  active_thread_ = t;

  debug_dump_state();
  lock.unlock();
  cv_.notify_all();
}

inline void scheduler::pass_control_to_once_viable(thread_impl* t) {
  MIT_LOG("Waiting for thread {} to become viable", t->id);
  while (t->activated && !t->viable()) {
    yield_control();
  }
  if (t->viable()) {
    MIT_LOG("Thread {} is now viable, passing control to it", t->id);
    pass_control_to(t);
    MIT_LOG("Passed control to thread {}", t->id);
  } else {
    MIT_LOG("Wait complete for thread {}, it is no longer viable", t->id);
  }
}

inline thread_impl* scheduler::get_next_thread() const {
  MIT_LOG("Getting next thread from viable threads");
  debug_dump_state();

  int next = boost_util::get_random_set_index(prng_, viable_threads_);
  MIT_LOG("Got next thread index: {}", next);
  if (next < 0) {
    return nullptr;
  }
  return all_threads_[next];
}

inline void scheduler::wait_on_helper(condition_variable* cv, unique_lock<mutex>& lock,
                                      predicate_t pred) {
  if (mid_orderly_shutdown_) {
    throw OrderlyShutdownException();
  }
  MIT_LOG("Thread {} waiting on condition variable {}", active_thread_->id, cv->id_);
  active_thread_->mark_as_blocked_by(cv);
  if (!cv_block_map_[cv->id_]) {
    cv_block_map_[cv->id_] = new thread_predicate_vec_t();
  }
  cv_block_map_[cv->id_]->push_back(std::make_pair(active_thread_, pred));
  viable_threads_[active_thread_->id] = false;
  debug_dump_state();
}

}  // namespace mit
