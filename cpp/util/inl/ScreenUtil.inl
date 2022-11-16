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
  /*
  if (!cur_term) {
    int result;
    setupterm(nullptr, STDOUT_FILENO, &result);
    if (result <= 0) return;
  }
  putp(tigetstr("clear"));
   */
}

}  // namespace util
