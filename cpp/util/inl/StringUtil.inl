#include <util/StringUtil.hpp>

#include <iostream>

#include <boost/algorithm/string.hpp>

#include <util/Exception.hpp>

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

inline float atof_safe(const std::string& s) {
  size_t read = 0;
  float f = std::stof(s, &read);
  if (read != s.size() || s.empty()) {
    throw util::Exception("atof failure %s(\"%s\")", __func__, s.c_str());
  }
  return f;
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

}  // namespace util
