#include <util/ScreenUtil.hpp>

#include <cstdlib>
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace util {

inline int get_screen_width() {
  static int width = 0;
  if (width == 0) {
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    width = w.ws_col;
  }
  return width;
}

/*
 * See: https://cplusplus.com/articles/4z18T05o/
 */
inline void clearscreen() {
  if (system("clear")) {
    throw std::exception();
  }
}

inline void ScreenClearer::clear_once() {
  ScreenClearer* s = instance();
  if (s->ready_) {
    clearscreen();
    s->ready_ = false;
  }
}

inline void ScreenClearer::reset() { instance()->ready_ = true; }

inline ScreenClearer* ScreenClearer::instance() {
  if (!instance_) {
    instance_ = new ScreenClearer();
  }
  return instance_;
}

}  // namespace util
