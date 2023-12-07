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

inline int ThreadSafePrinter::printf(const char* format, ...) {
  validate_lock();
  line_start_ = true;
  int ret = print_timestamp();
  va_list args;
  va_start(args, format);
  ret += vprintf(format, args);
  va_end(args);
  std::cout.flush();
  return ret;
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

inline ThreadSafePrinter& ThreadSafePrinter::operator<<(std_endl_t f) {
  validate_lock();
  f(std::cout);
  line_start_ = true;
  return *this;
}

inline void ThreadSafePrinter::validate_lock() const {
  if (!lock_.owns_lock()) {
    throw std::runtime_error("ThreadSafePrinter: lock not held");
  }
}

inline int ThreadSafePrinter::print_timestamp() const {
  int out = 0;
  if (print_timestamp_) {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);

    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&now_c));

    auto now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()) % 1000000000LL;
    out += ::printf("%s.%09d ", buf, (int)now_ns.count());
  }

  std::string s(thread_id_ * kWhitespacePrefixLength, ' ');
  out += ::printf("%s", s.c_str());
  return out;
}

}  // namespace util
