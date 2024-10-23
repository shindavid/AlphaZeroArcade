#include <util/StringUtil.hpp>

#include <util/Exception.hpp>

#include <algorithm>
#include <iostream>
#include <regex>
#include <sstream>

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

inline std::vector<std::string> split(const std::string& s, const char* t) {
  std::vector<std::string> result;

  std::string_view t_view(t);
  if (t_view.empty()) {
    // Split on any whitespace (like s.split() in Python)
    std::istringstream stream(s);
    std::string word;
    while (stream >> word) {
      result.push_back(word);
    }
  } else {
    // Split on the specified separator (like s.split(t) in Python)
    std::string::size_type start = 0;
    std::string::size_type end;

    while ((end = s.find(t_view, start)) != std::string::npos) {
      result.push_back(s.substr(start, end - start));
      start = end + t_view.length();
    }

    // Add the last part
    result.push_back(s.substr(start));
  }

  return result;
}

inline std::vector<std::string> splitlines(const std::string &s) {
  std::vector<std::string> result;
  std::string::size_type start = 0;
  std::string::size_type end;

  while ((end = s.find('\n', start)) != std::string::npos) {
    result.push_back(s.substr(start, end - start));
    start = end + 1;
  }

  if (start < s.size()) {
    result.push_back(s.substr(start));
  }

  return result;
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

inline bool ends_with(const std::string& value, const std::string& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline size_t terminal_width(const std::string& str) {
  // This regular expression matches ANSI escape sequences
  std::regex ansi_escape("\033\\[[0-9;]*m");

  // Remove all escape sequences using regex
  std::string cleaned_str = std::regex_replace(str, ansi_escape, "");

  return cleaned_str.size();  // Return the size of the cleaned string
}

}  // namespace util
