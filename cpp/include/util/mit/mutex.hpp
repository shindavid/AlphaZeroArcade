#pragma once

#ifndef MIT_TEST_MODE
static_assert(false, "This file is not intended to be #include'd directly.");
#endif  // MIT_TEST_MODE

namespace mit {

// Drop-in replacement for std::mutex that can be used in unit tests.
//
// Not all std::mutex functionality is implemented in this class. Only those methods that are
// used in the AlphaZeroArcade codebase are provided. In the future, we can extend this class to
// include more functionality as needed.
class mutex {
 public:
  friend class scheduler;

  mutex();
  ~mutex();

  mutex(const mutex&) = delete;
  mutex& operator=(const mutex&) = delete;

  void lock();
  void unlock();

 private:
  int id_ = -1;  // set by scheduler
  bool locked_ = false;
};

}  // namespace mit

#include <inline/util/mit/mutex.inl>
