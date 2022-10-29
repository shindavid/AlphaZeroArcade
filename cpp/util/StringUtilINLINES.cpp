#include <util/StringUtil.hpp>

#include <boost/algorithm/string.hpp>

namespace util {

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
