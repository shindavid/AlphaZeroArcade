#pragma once

#include <mutex>
#include <string>

namespace util {

/*
 * Allows for std::cout << and printf() style printing, but in a thread-safe manner. Directly using std::cout
 * can lead to interleaved output from different threads. Directly using printf() is safe from this concern, but
 * using multiple printf() calls in a row can lead to interleaved output from different threads.
 *
 * Usage:
 *
 * util::ThreadSafePrinter p(my_thread_id);
 * p.printf("foo %d\n", 3);
 * p << "bar " << 4;
 * p.endl();  // ghetto std::endl, supporting << std::endl is a pain
 * p.release();
 *
 * Under the hood, the ThreadSafePrinter constructor grabs a singleton mutex, ensuring thread-safety. The mutex is
 * released via the release() method, or in the destructor if release() is never invoked explicitly.
 *
 * An int thread-id can be passed as an optional constructor argument. All printed output lines will be prefixed with
 * whitespace whose length is proportional to the thread-id.
 *
 * Finally, each line is optionally prefixed with a timestamp (before the thread whitespace). This is enabled by
 * default, but can be disabled by passing false as the second constructor argument.
 */

class ThreadSafePrinter {
public:
  static const int kWhitespacePrefixLength = 50;
  ThreadSafePrinter(int thread_id = 0, bool print_timestamp = true);
  ~ThreadSafePrinter();

  void release();
  int printf(const char* fmt, ...) __attribute__((format(printf, 2, 3)));  // assumes fmt ends with '\n'
  template<typename T> ThreadSafePrinter& operator<<(const T& t);
  void endl();

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

#include <util/inl/ThreadSafePrinter.inl>
