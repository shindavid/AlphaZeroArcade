#pragma once

#include <util/Exceptions.hpp>

#include <format>
#include <source_location>

/*
 * A variety of assert functions:
 *
 * - DEBUG_ASSERT() - throws a util::Exception if the condition is false; enabled only for debug
 *   builds.
 *
 * - RELEASE_ASSERT() - throws a util::Exception if the condition is false; enabled for debug AND
 *   release builds.
 *
 * - CLEAN_ASSERT() - throws a util::CleanException if the condition is false; enabled for debug
 *   AND release builds. See util::CleanException documentation.
 *
 * Each variant can be passed a single bool, or a bool followed by a format string and
 * additional formatting arguments.
 *
 * Why don't we use the standard assert() function? There are a few reasons:
 *
 * 1. There is a gcc bug that causes spurious assert() failures deep in the eigen3 library. So
 *    we don't want to enable assert()'s, even in debug builds.
 *
 * 2. When disabled, the assert() macro compiles out the arguments, which is undesirable for a
 *    couple reasons:
 *
 *    A. If there is a bug in the arguments, we won't know until we compile with asserts enabled.
 *    B. If a local variable is declared and then used only in an assert(), the compiler will
 *       complain that the variable is unused, which is annoying.
 *
 * Our *ASSERT() macros overcome these issues, while incurring zero runtime-cost for disabled
 * *ASSERT()'s. In other words, the arguments are only evaluated if the assertion is enabled.
 */

#define DEBUG_ASSERT(COND, ...)                                                                    \
  do {                                                                                             \
    if (IS_DEFINED(DEBUG_BUILD)) {                                                                 \
      util::detail::assert_impl<util::DebugAssertionError>(#COND, std::source_location::current(), \
                                                           COND, ##__VA_ARGS__);                   \
    }                                                                                              \
  } while (0)

#define RELEASE_ASSERT(COND, ...)                                                                  \
  do {                                                                                             \
    util::detail::assert_impl<util::ReleaseAssertionError>(#COND, std::source_location::current(), \
                                                           COND, ##__VA_ARGS__);                   \
  } while (0)

#define CLEAN_ASSERT(COND, ...)                                                                  \
  do {                                                                                           \
    util::detail::assert_impl<util::CleanAssertionError>(#COND, std::source_location::current(), \
                                                         COND, ##__VA_ARGS__);                   \
  } while (0)

namespace util {
namespace detail {

template <typename ExceptionT, typename... Ts>
inline void assert_impl([[maybe_unused]] const char* cond_str, const std::source_location& loc,
                        bool cond, const std::format_string<Ts...>& fmt, Ts&&... ts) {
  if (!cond) {
    throw ExceptionT("{} failed: {} [{}:{}]", ExceptionT::descr(),
                     std::format(fmt, std::forward<Ts>(ts)...), loc.file_name(), loc.line());
  }
}

template <typename ExceptionT, typename... Ts>
inline void assert_impl(const char* cond_str, const std::source_location& loc, bool cond) {
  if (!cond) {
    throw ExceptionT("{} failed: {} [{}:{}]", ExceptionT::descr(), cond_str, loc.file_name(),
                     loc.line());
  }
}

}  // namespace detail
}  // namespace util
