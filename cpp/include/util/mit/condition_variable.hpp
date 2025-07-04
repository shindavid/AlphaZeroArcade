#pragma once

#include <util/mit/mutex.hpp>
#include <util/mit/unique_lock.hpp>

#include <condition_variable>

#ifndef MIT_TEST_MODE
static_assert(false, "This file is not intended to be #include'd directly.");
#endif  // MIT_TEST_MODE

namespace mit {

// Drop-in replacement for std::condition_variable that can be used in unit tests.
//
// Not all std::condition_variable functionality is implemented in this class. Only those methods
// that are used in the AlphaZeroArcade codebase are provided. In the future, we can extend this
// class to include more functionality as needed.
class condition_variable {
 public:
  condition_variable();
  ~condition_variable();

  condition_variable(const condition_variable&) = delete;
  condition_variable& operator=(const condition_variable&) = delete;

  void notify_one();
  void notify_all();

  void wait(mit::unique_lock<mit::mutex>& lock);

  template <class Predicate>
  void wait(mit::unique_lock<mit::mutex>& lock, Predicate pred);

 private:
  std::condition_variable cv_;
};

}  // namespace mit

#include <inline/util/mit/condition_variable.inl>
