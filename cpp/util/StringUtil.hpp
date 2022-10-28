#pragma once

/*
 * Various string utilities
 */

#include <cstdarg>
#include <string>
#include <vector>

#include <util/Exception.hpp>

namespace util {

/*
 * split(s) behaves just like s.split() in python.
 *
 * TODO: suppose split(s, t), to behave like s.split(t).
 */
std::vector<std::string> split(const std::string& s);

/*
 * Like sprintf(), but conveniently bypasses need to declare the char buffer.
 *
 * The template parameter N dictates the char buffer size. In case of overflow, throws an exception.
 */
template<int N=1024>
inline std::string create_string(char const* fmt, ...) __attribute__((format(printf, 1, 2))) {
  char text[N];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(text, sizeof(text), fmt, ap);
  va_end(ap);

  if (n < 0) {
    throw Exception("create_string<%d>(): encountered encoding error (fmt=%s)", N, fmt);
  }
  if (n >= N) {
    throw Exception("create_string<%d>(): char buffer overflow (%d)", N, n);
  }
  return std::string(text);
}

}  // namespace util

#include <util/StringUtilINLINES.cpp>
