#include "util/ScreenUtil.hpp"

#include "util/Exceptions.hpp"

#include <sys/ioctl.h>

#include <cstdlib>
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
    throw util::Exception("Failed to clear screen: system() returned non-zero exit status");
  }
}

}  // namespace util
