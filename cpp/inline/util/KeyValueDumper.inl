#include <util/KeyValueDumper.hpp>

#include <util/LoggingUtil.hpp>
#include <util/StringUtil.hpp>

namespace util {

inline void KeyValueDumper::add(const std::string& key, const char* value_fmt, ...) {
  constexpr int N = 1024;
  char value[N];
  va_list ap;
  va_start(ap, value_fmt);
  int n = vsnprintf(value, sizeof(value), value_fmt, ap);
  va_end(ap);

  if (n < 0) {
    throw Exception("KeyValueDumper::add(): encountered encoding error (N=%d, fmt=\"%s\")", N,
                    value_fmt);
  }
  if (n >= N) {
    throw Exception("KeyValueDumper::add(): char buffer overflow (%d >= %d)", n, N);
  }

  instance().vec_.emplace_back(std::format("{}:", key), value);
}

inline void KeyValueDumper::flush() {
  int max_key_len = 0;
  int max_value_len = 0;
  for (const auto& p : instance().vec_) {
    max_key_len = std::max(max_key_len, (int)p.first.size());
    max_value_len = std::max(max_value_len, (int)p.second.size());
  }

  for (const auto& p : instance().vec_) {
    LOG_INFO("{:<{}} {:>{}}", p.first, max_key_len, p.second, max_value_len);
  }
  instance().vec_.clear();
  std::cout.flush();
}

inline KeyValueDumper& KeyValueDumper::instance() {
  static KeyValueDumper instance;
  return instance;
}

}  // namespace util
