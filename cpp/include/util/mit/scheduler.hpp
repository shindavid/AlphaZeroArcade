#pragma once

#include <util/mit/unique_lock.hpp>

#include <boost/dynamic_bitset.hpp>

#include <condition_variable>
#include <map>
#include <mutex>
#include <vector>

namespace mit {

class condition_variable;
class mutex;
class thread;

class scheduler {
 public:
  static scheduler* instance();

  void register_thread(thread*);
  void unregister_thread(thread*);
  void pass_control_to(thread*);
  void switch_active_thread();
  thread* active_thread();
  void move_thread(thread* from, thread* to);
  void deactivate_thread(thread* t);

  void register_mutex(mutex*);
  void unregister_mutex(mutex*);
  void mark_active_thread_as_blocked_by(mutex* m);
  bool is_locked(mutex* m) const;
  void mark_as_locked(mutex* m);
  void mark_as_unlocked(mutex* m);

  void register_condition_variable(condition_variable*);
  void unregister_condition_variable(condition_variable*);
  void notify_one(condition_variable*);
  void notify_all(condition_variable*);
  void wait_on(condition_variable*, unique_lock<mutex>& lock);

  template <class Predicate>
  void wait_on(condition_variable*, unique_lock<mutex>& lock, Predicate pred);

 private:
  scheduler();
  void validate_thread(thread*) const;
  void validate_thread_viability(thread*) const;
  void validate_mutex(mutex*) const;

  using thread_vec_t = std::vector<thread*>;
  using mutex_vec_t = std::vector<mutex*>;
  using mutex_block_map_t = std::map<mutex*, thread_vec_t>;

  std::condition_variable cv_;
  std::mutex mutex_;

  mutex_block_map_t mutex_block_map_;

  mutex_vec_t all_mutexes_;
  boost::dynamic_bitset<> locked_mutexes_;  // Bitset to track which mutexes are locked

  thread_vec_t all_threads_;
  boost::dynamic_bitset<> viable_threads_;  // Bitset to track which threads are viable
  thread* active_thread_ = nullptr;

  static scheduler* instance_;
};

}  // namespace mit

#include <inline/util/mit/scheduler.inl>
