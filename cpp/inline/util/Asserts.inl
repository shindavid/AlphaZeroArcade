#include <util/Asserts.hpp>

#include <util/CppUtil.hpp>
#include <util/Exception.hpp>

#include <cstdarg>

namespace util {

inline void debug_assert(bool condition) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;
  if (condition) return;
  throw Exception();
}

inline void release_assert(bool condition) {
  if (condition) return;
  throw Exception();
}

inline void clean_assert(bool condition) {
  if (condition) return;
  throw CleanException();
}

template <typename... Ts>
void debug_assert(bool condition, std::format_string<Ts...> fmt, Ts&&... ts) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;
  if (condition) return;
  throw Exception(fmt, std::forward<Ts>(ts)...);
}

template <typename... Ts>
void release_assert(bool condition, std::format_string<Ts...> fmt, Ts&&... ts) {
  if (condition) return;
  throw Exception(fmt, std::forward<Ts>(ts)...);
}

template <typename... Ts>
void clean_assert(bool condition, std::format_string<Ts...> fmt, Ts&&... ts) {
  if (condition) return;
  throw CleanException(fmt, std::forward<Ts>(ts)...);
}

}  // namespace util
