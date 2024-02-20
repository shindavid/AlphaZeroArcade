#pragma once

#include <iostream>
#include <mutex>
#include <string>

namespace util {

/*
 * A class for generating a timestamp prefix for logging. The prefix is of the form
 * "HH:MM:SS.NNNNNNNNN".
 */
class TimestampPrefix {
 public:
  static const char* get();

 private:
  static constexpr int buf_size = 32;
  static TimestampPrefix* instance();
  static TimestampPrefix* instance_;

  char buf_[buf_size];
};

/*
 * Allows for std::cout << and printf() style printing, but in a thread-safe manner. Directly using
 * std::cout can lead to interleaved output from different threads. Directly using printf() is safe
 * from this concern, but using multiple printf() calls in a row can lead to interleaved output from
 * different threads.
 *
 * Usage:
 *
 * util::ThreadSafePrinter p(my_thread_id);
 * p.printf("foo %d\n", 3);
 * p << "bar " << 4 << std::endl;
 * p.release();  // optional, called in destructor if not invoked explicitly
 *
 * Under the hood, the ThreadSafePrinter constructor grabs a singleton mutex, ensuring
 * thread-safety. The mutex is released via the release() method, or in the destructor if release()
 * is never invoked explicitly.
 *
 * An int thread-id can be passed as an optional constructor argument. All printed output lines will
 * be prefixed with whitespace whose length is proportional to the thread-id.
 *
 * Finally, each line is optionally prefixed with a timestamp (before the thread whitespace). This
 * is enabled by default, but can be disabled by passing false as the second constructor argument.
 *
 * TODO: once the std::format library is implemented in gcc, use that instead of printf-style
 * formatting.
 */

class ThreadSafePrinter {
 public:
  static const int kWhitespacePrefixLength = 50;
  ThreadSafePrinter(int thread_id = 0, bool print_timestamp = true);
  ~ThreadSafePrinter();

  void release();
  int printf(const char* fmt, ...) __attribute__((format(printf, 2, 3)));
  template <typename T>
  ThreadSafePrinter& operator<<(const T& t);

  // std::endl support
  using std_endl_t = std::ostream& (*)(std::ostream&);
  ThreadSafePrinter& operator<<(std_endl_t);

 private:
  void validate_lock() const;
  int print_timestamp() const;
  static std::mutex mutex_;

  const int thread_id_;
  const bool print_timestamp_;
  std::unique_lock<std::mutex> lock_;
  bool line_start_ = true;
};

}  // namespace util

#include <inline/util/LoggingUtil.inl>
