#include <util/KeyValueDumper.hpp>

#include <util/LoggingUtil.hpp>
#include <util/StringUtil.hpp>

namespace util {

KeyValueDumper* KeyValueDumper::instance_ = nullptr;

void KeyValueDumper::add(const std::string& key, const char* value_fmt, ...) {
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

  instance()->vec_.emplace_back(util::create_string("%s:", key.c_str()), value);
}

void KeyValueDumper::flush() {
  int max_key_len = 0;
  int max_value_len = 0;
  for (const auto& p : instance()->vec_) {
    max_key_len = std::max(max_key_len, (int)p.first.size());
    max_value_len = std::max(max_value_len, (int)p.second.size());
  }

  std::string fmt_str = util::create_string("%%-%ds %%%ds\n", max_key_len, max_value_len);
  const char* fmt = fmt_str.c_str();
  for (const auto& p : instance()->vec_) {
    LOG_INFO << util::create_string(fmt, p.first.c_str(), p.second.c_str());
  }
  instance()->vec_.clear();
  std::cout.flush();
}

KeyValueDumper* KeyValueDumper::instance() {
  if (instance_ == nullptr) {
    instance_ = new KeyValueDumper();
  }
  return instance_;
}

}  // namespace util
