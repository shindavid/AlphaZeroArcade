#include <util/ScreenUtil.hpp>

#include <cstdlib>
#include <unistd.h>
#include <term.h>

namespace util {

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
