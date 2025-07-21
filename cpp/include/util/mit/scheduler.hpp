#pragma once

#include "util/Exceptions.hpp"
#include "util/mit/exceptions.hpp"
#include "util/mit/id_provider.hpp"
#include "util/mit/unique_lock.hpp"

#include <boost/dynamic_bitset.hpp>

#include <condition_variable>
#include <exception>
#include <mutex>
#include <random>
#include <vector>

namespace mit {

class condition_variable;
class mutex;
class thread;
class thread_impl;

// Singleton class used to coordinate thread execution in a single-threaded manner.
//
// For the most part, scheduler is thread-safe. The only sensitivity surrounds the use of methods
// invoked inside the mit::thread constructor.
class scheduler {
 public:
  scheduler(const scheduler&) = delete;
  scheduler& operator=(const scheduler&) = delete;
  ~scheduler() noexcept(false);

  static scheduler& instance();
  void seed(int s) { prng_.seed(s); }
  void reset();

  void register_thread(thread_impl*);
  void unregister_thread(thread_impl*);
  thread_impl* active_thread();
  void join_thread(thread_impl* t);

  void yield_control(thread_impl* t = nullptr);  // Beware thread-safety
  void deactivate_thread(thread_impl* t);        // Beware thread-safety
  void block_until_has_control(thread_impl* t);  // Beware thread-safety

  void register_mutex(mutex* m);
  void unregister_mutex(mutex* m);
  void lock_mutex(mutex* m);
  void unlock_mutex(mutex* m);
  void mark_active_thread_as_blocked_by(mutex* m);
  void mark_as_unlocked(mutex* m);

  void register_condition_variable(condition_variable*);
  void unregister_condition_variable(condition_variable*);
  void notify_one(condition_variable*);
  void notify_all(condition_variable*);
  void wait_on(condition_variable*, unique_lock<mutex>& lock);

  template <class Predicate>
  void wait_on(condition_variable*, unique_lock<mutex>& lock, Predicate pred);

  void enable_bug_catching_mode();
  void disable_bug_catching_mode();
  void handle_bug_detected_error(const BugDetectedError&);
  void throw_bug_detected_error(const char* func);
  void commence_orderly_shutdown();
  void handle_exception();
  bool mid_orderly_shutdown() const { return mid_orderly_shutdown_; }

 private:
  using index_vec_t = std::vector<int>;
  using predicate_t = std::function<bool()>;
  using thread_map_t = std::vector<thread_impl*>;
  using thread_vec_t = std::vector<thread_impl*>;
  using mutex_block_map_t = std::vector<thread_vec_t*>;

  using thread_predicate_pair_t = std::pair<thread_impl*, predicate_t>;
  using thread_predicate_vec_t = std::vector<thread_predicate_pair_t>;
  using cv_block_map_t = std::vector<thread_predicate_vec_t*>;

  scheduler();

  void init_main_thread();
  void debug_dump_state() const;
  void dump_state() const;
  void pass_control_to(thread_impl*);
  void pass_control_to_once_viable(thread_impl*);
  thread_impl* get_next_thread() const;
  void wait_on_helper(condition_variable* cv, unique_lock<mutex>& lock, predicate_t pred);

  mutable std::mt19937 prng_;
  std::condition_variable cv_;
  std::mutex mutex_;

  id_provider thread_id_provider_;
  thread_vec_t all_threads_;
  boost::dynamic_bitset<> viable_threads_;  // Bitset to track which threads are viable
  thread* main_thread_ = nullptr;
  thread_impl* active_thread_ = nullptr;

  id_provider mutex_id_provider_;
  mutex_block_map_t mutex_block_map_;

  id_provider cv_id_provider_;
  cv_block_map_t cv_block_map_;

  thread_predicate_vec_t tmp_thread_predicate_vec_;  // Used to avoid dynamic allocation

  std::exception_ptr caught_exception_;  // Used to catch exceptions in unit tests
  int uncaught_throw_count_ = 0;
  bool bug_catching_enabled_ = false;
  bool mid_orderly_shutdown_ = false;
};

}  // namespace mit
