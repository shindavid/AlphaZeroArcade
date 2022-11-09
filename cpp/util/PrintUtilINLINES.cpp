#include <util/PrintUtil.hpp>

namespace util {

namespace detail {

/*
 * Not to be accessed directly.
 */
class _xprintf_helper {
public:
  void set_xprintf_target(std::ostringstream& target) {
    target_ = &target;
    if (!buf_) {
      buf_size_ = 1024;
      buf_ = new char[buf_size_];
    }
  }

  void clear_xprintf_target() {
    target_ = nullptr;
  }

  int xprintf(char const* fmt, ...) __attribute__((format(printf, 2, 3))) {
    if (target_) {
      int n = buf_size_;
      for (; n >= buf_size_; resize(2 * n)) {
        va_list ap;
        va_start(ap, fmt);
        n = snprintf(buf_, buf_size_, fmt, ap);
        va_end(ap);
      }
      return n;
    } else {
      va_list ap;
      va_start(ap, fmt);
      int n = printf(fmt, ap);
      va_end(ap);
      return n;
    }
  }

  static _xprintf_helper& instance() { return instance_; }

private:
  void resize(int buf_size) {
    delete[] buf_;
    buf_size_ = buf_size;
    buf_ = new char[buf_size_];
  }

  static _xprintf_helper instance_;
  char* buf_ = nullptr;
  int buf_size_ = 0;
  std::ostringstream* target_ = nullptr;
};

}  // namespace detail

inline void set_xprintf_target(std::ostringstream& target) {
  detail::_xprintf_helper::instance().set_xprintf_target(target);
}

inline void clear_xprintf_target() {
  detail::_xprintf_helper::instance().clear_xprintf_target();
}

inline int xprintf(char const* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = detail::_xprintf_helper::instance().xprintf(fmt, ap);
  va_end(ap);
  return n;
}

}  // namespace util
