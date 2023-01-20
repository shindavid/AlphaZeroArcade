#include <util/StringUtil.hpp>

#include <iostream>

#include <boost/algorithm/string.hpp>

namespace util {

namespace detail {

/*
 * Adapted from https://stackoverflow.com/a/48896410/543913
 */
template <typename Str>
constexpr uint64_t str_hash(const Str& toHash) {
  uint64_t result = 0xcbf29ce484222325; // FNV offset basis

  for (char c : toHash) {
    result ^= c;
    result *= 1099511628211; // FNV prime
  }

  return result;
}

}  // namespace detail

inline constexpr uint64_t str_hash(const char* c) {
  return detail::str_hash(std::string_view(c));
}

inline std::vector<std::string> split(const std::string &s) {
  namespace ba = boost::algorithm;

  std::string delims = " \n\t\r\v\f";  // matches python isspace()
  std::vector<std::string> tokens;
  ba::split(tokens, s, boost::is_any_of(delims));
  return tokens;
}

template<int N>
inline std::string create_string(char const *fmt, ...) {
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

template<typename T>
void param_dump(const char* descr, const char* param_fmt, T param) {
  // TODO: fix misalignment when param_fmt has trailing chars (like units)
  if (false) {
    printf(param_fmt, param);
  }
  constexpr int descr_width = 40;
  constexpr int param_width = 10;

  // "%-50s %10d\n" -> "X-%ds X%d%s\n"
  std::string fmt = create_string("%%-%ds %%%d%s\n", descr_width, param_width, param_fmt + 1);
  printf(fmt.c_str(), create_string("%s:", descr).c_str(), param);
}

}  // namespace util
