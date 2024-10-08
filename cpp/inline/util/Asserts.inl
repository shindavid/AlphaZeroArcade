#include <util/Asserts.hpp>

#include <util/CppUtil.hpp>
#include <util/Exception.hpp>

#include <cstdarg>
#include <iostream>

namespace util {

inline void debug_assert(bool condition, char const* fmt, ...) {
  if (!IS_MACRO_ENABLED(DEBUG_BUILD)) return;
  if (condition) return;
  if (!fmt) throw Exception();
  va_list ap;
  va_start(ap, fmt);
  char text[1024];
  vsnprintf(text, sizeof(text), fmt, ap);
  va_end(ap);
  throw Exception("%s", text);
}

inline void release_assert(bool condition, char const* fmt, ...) {
  if (condition) return;
  if (!fmt) throw Exception();
  va_list ap;
  va_start(ap, fmt);
  char text[1024];
  vsnprintf(text, sizeof(text), fmt, ap);
  va_end(ap);
  throw Exception("%s", text);
}

inline void clean_assert(bool condition, char const* fmt, ...) {
  if (condition) return;
  if (!fmt) throw CleanException();
  va_list ap;
  va_start(ap, fmt);
  char text[1024];
  vsnprintf(text, sizeof(text), fmt, ap);
  va_end(ap);
  throw CleanException("%s", text);
}

}  // namespace util
