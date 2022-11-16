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

  int xprintf(char const* fmt, va_list* args) {
    if (target_) {
      int n = buf_size_;
      for (; n >= buf_size_; resize(2 * n)) {
        n = vsnprintf(buf_, buf_size_, fmt, *args);
      }
      (*target_) << buf_;
      return n;
    } else {
      return vprintf(fmt, *args);
    }
  }

  void xflush() {
    if (!target_) {
      std::cout.flush();
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
  va_list args;
  va_start(args, fmt);
  int n = detail::_xprintf_helper::instance().xprintf(fmt, &args);
  va_end(args);
  return n;
}

inline void xflush() {
  detail::_xprintf_helper::instance().xflush();
}

}  // namespace util
