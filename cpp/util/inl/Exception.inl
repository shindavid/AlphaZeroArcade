#include <util/Exception.hpp>

namespace util {

inline Exception::Exception(char const* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(text_, sizeof(text_), fmt, ap);
  va_end(ap);
}

inline CleanException::CleanException(char const* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(text_, sizeof(text_), fmt, ap);
  va_end(ap);
}

inline void clean_assert(bool condition, char const* fmt, ...) {
  if (condition) return;
  va_list ap;
  va_start(ap, fmt);
  char text[1024];
  vsnprintf(text, sizeof(text), fmt, ap);
  va_end(ap);
  throw CleanException("%s", text);
}

}  // namespace util
