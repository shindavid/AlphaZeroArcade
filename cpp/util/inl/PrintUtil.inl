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
      int n = 0;
      while (true) {
        va_list args_copy;
        va_copy(args_copy, *args);
        n = vsnprintf(buf_, buf_size_, fmt, args_copy);
        va_end(args_copy);
        if (n >= buf_size_) {
          resize(2 * n);
          continue;
        }
        break;
      }
      (*target_) << buf_;
      return n;
    } else {
      int n = vprintf(fmt, *args);
      return n;
    }
  }

  void xflush() {
    if (!target_) {
      std::cout.flush();
    }
  }

  static _xprintf_helper& instance() { return instance_; }

  static void debug_dump() {
    std::cout << "_xprintf_helper::target_: " << instance_.target_ << std::endl;
    if (instance_.target_) {
      std::cout << "BEGIN(" << instance_.target_->str().size() << ")" << std::endl;
      std::cout << instance_.target_->str() << std::endl;
      std::cout << "END" << std::endl;
    }
  }

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
