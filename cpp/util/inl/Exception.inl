#include <util/Exception.hpp>

namespace util {

inline Exception::Exception(char const* fmt, ...) {
  if (fmt) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(text_, sizeof(text_), fmt, ap);
    va_end(ap);
  } else {
    text_[0] = 0;
  }
}

inline CleanException::CleanException(char const* fmt, ...) {
  if (fmt) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(text_, sizeof(text_), fmt, ap);
    va_end(ap);
  } else {
    text_[0] = 0;
  }
}

}  // namespace util
