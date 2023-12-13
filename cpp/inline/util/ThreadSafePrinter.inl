#include <util/ThreadSafePrinter.hpp>

#include <chrono>
#include <ctime>
#include <iostream>

namespace util {

inline ThreadSafePrinter::ThreadSafePrinter(int thread_id, bool print_timestamp)
    : thread_id_(thread_id), print_timestamp_(print_timestamp), lock_(mutex_) {}

inline ThreadSafePrinter::~ThreadSafePrinter() { release(); }

inline void ThreadSafePrinter::release() {
  if (lock_.owns_lock()) {
    lock_.unlock();
  }
}

template <typename T>
inline ThreadSafePrinter& ThreadSafePrinter::operator<<(const T& t) {
  validate_lock();
  if (line_start_) {
    print_timestamp();
    line_start_ = false;
  }
  std::cout << t;
  return *this;
}

inline void ThreadSafePrinter::validate_lock() const {
  if (!lock_.owns_lock()) {
    throw std::runtime_error("ThreadSafePrinter: lock not held");
  }
}

}  // namespace util
