#pragma once

#include <string>
#include <utility>
#include <vector>

namespace util {

class KeyValueDumper {
 public:
  /*
   * Adds a key-value pair to dump.
   *
   * add("foo", "%.3fsec", 1.2345);
   * add("some_really_strong_str", "%d", 10);
   *
   * will result in the following output:
   *
   * foo:                    1.235sec
   * some_really_strong_str:       10
   *
   * Note:
   *
   * - The addition of colon and \n characters
   * - The right-alignment of the values
   */
  static void add(const std::string& key, const char* value_fmt, ...)
      __attribute__((format(printf, 2, 3)));

  /*
   * Dumps all add()'ed parameters to stdout.
   */
  static void flush();

 private:
  static KeyValueDumper* instance();

  static KeyValueDumper* instance_;

  using pair_t = std::pair<std::string, std::string>;
  using vec_t = std::vector<pair_t>;

  vec_t vec_;
};

}  // namespace util

#include <inline/util/KeyValueDumper.inl>
