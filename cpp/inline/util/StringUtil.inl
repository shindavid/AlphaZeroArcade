#include "util/StringUtil.hpp"

#include <algorithm>
#include <regex>

namespace util {

namespace detail {

/*
 * Adapted from https://stackoverflow.com/a/48896410/543913
 */
template <typename Str>
constexpr uint64_t str_hash(const Str& toHash) {
  uint64_t result = 0xcbf29ce484222325;  // FNV offset basis

  for (char c : toHash) {
    result ^= c;
    result *= 1099511628211;  // FNV prime
  }

  return result;
}

}  // namespace detail

inline constexpr uint64_t str_hash(const char* c) { return detail::str_hash(std::string_view(c)); }

inline float atof_safe(const std::string& s) {
  size_t read = 0;
  float f = std::stof(s, &read);
  if (read != s.size() || s.empty()) {
    throw std::invalid_argument(std::format("atof failure {}(\"{}\")", __func__, s));
  }
  return f;
}

inline std::vector<std::string> split(const std::string& s, const char* t) {
  std::vector<std::string> result;
  split(result, s, t);
  return result;
}

inline int split(std::vector<std::string>& result, const std::string& s, const char* t) {
  std::string_view sep(t);
  std::size_t token_count = 0;

  if (sep.empty()) {
    std::string_view sv(s);
    std::size_t pos = 0, n = sv.size();
    while (pos < n) {
      while (pos < n && std::isspace(static_cast<unsigned char>(sv[pos]))) ++pos;
      if (pos >= n) break;
      std::size_t start = pos;
      while (pos < n && !std::isspace(static_cast<unsigned char>(sv[pos]))) ++pos;
      std::string_view tok = sv.substr(start, pos - start);

      if (token_count < result.size())
        result[token_count] = tok;
      else
        result.emplace_back(tok);

      ++token_count;
    }
  } else {
    std::size_t start = 0, end;
    while ((end = s.find(sep, start)) != std::string::npos) {
      std::string_view tok(s.data() + start, end - start);
      if (token_count < result.size())
        result[token_count] = tok;
      else
        result.emplace_back(tok);
      ++token_count;
      start = end + sep.size();
    }
    // last segment
    std::string_view tok(s.data() + start, s.size() - start);
    if (token_count < result.size())
      result[token_count] = tok;
    else
      result.emplace_back(tok);
    ++token_count;
  }

  return int(token_count);
}

inline std::vector<std::string> splitlines(const std::string& s) {
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

inline bool ends_with(const std::string& value, const std::string& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline size_t terminal_width(const std::string& str) {
  // Ignore ANSI escape sequences for coloring and such
  static const std::regex ansi_regex("\033\\[[0-9;]*m");
  std::string cleaned_str = std::regex_replace(str, ansi_regex, "");

  // Here, we assume that all unicode characters are 1 character wide. In reality, some unicode
  // characters like some emojis and various East Asian glyphs can have more than 1 character. But
  // we aren't using any such characters currently, and are unlikely to going forward. A more
  // accurate solution would require more sophistication and be costlier. So we go with this
  // simpler hack for now.
  static const std::regex unicode_regex(
    "[\\xC2-\\xDF][\\x80-\\xBF]"      // 2-byte sequence
    "|[\\xE0-\\xEF][\\x80-\\xBF]{2}"  // 3-byte sequence
    "|[\\xF0-\\xF4][\\x80-\\xBF]{3}"  // 4-byte sequence
  );
  cleaned_str = std::regex_replace(cleaned_str, unicode_regex, "?");

  return cleaned_str.size();
}

inline std::string grammatically_join(const std::vector<std::string>& items,
                                      const std::string& conjunction, bool oxford_comma) {
  if (items.empty()) return "";
  if (items.size() == 1) return items[0];
  if (items.size() == 2) {
    return items[0] + " " + conjunction + " " + items[1];
  }

  std::string result;
  for (size_t i = 0; i < items.size(); ++i) {
    result += items[i];
    if (i == items.size() - 2) {
      result += (oxford_comma ? ", " : " ") + conjunction + " ";
    } else if (i < items.size() - 1) {
      result += ", ";
    }
  }
  return result;
}

inline uint64_t parse_bytes(const std::string& str) {
  std::string num;
  std::string suf;

  bool has_decimal = false;

  for (char c : str) {
    bool decimal = c == '.';
    has_decimal |= decimal;
    if (std::isdigit(c) || decimal) {
      num.push_back(c);
    } else {
      suf.push_back(std::tolower(c));
    }
  }

  uint64_t mul = 1;
  if (suf == "kb" || suf == "kib")
    mul = 1ull << 10;
  else if (suf == "mb")
    mul = 1000ull * 1000;
  else if (suf == "mib")
    mul = 1ull << 20;
  else if (suf == "gb")
    mul = 1000ull * 1000 * 1000;
  else if (suf == "gib")
    mul = 1ull << 30;

  if (has_decimal) {
    double val = std::stod(num);
    return (uint64_t)(val * mul);
  } else {
    return std::stoull(num) * mul;
  }
}

}  // namespace util
