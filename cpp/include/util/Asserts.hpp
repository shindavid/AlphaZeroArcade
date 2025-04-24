#pragma once

#include <format>

/*
 * A variety of assert functions:
 *
 * - debug_assert() - throws a util::Exception if the condition is false; enabled only for debug
 *   builds.
 *
 * - release_assert() - throws a util::Exception if the condition is false; enabled for debug AND
 *   release builds.
 *
 * - clean_assert() - throws a util::CleanException if the condition is false; enabled for debug
 *   AND release builds. See util::CleanException documentation.
 *
 * NOTE: we don't use the standard assert() function because we set the macro NDEBUG in both
 * debug and release builds. This is because a gcc bug causes spurious assert() failures deep in
 * the eigen3 library.
 */
namespace util {

void debug_assert(bool condition);
void clean_assert(bool condition);
void release_assert(bool condition);

template <typename... Ts>
void debug_assert(bool condition, std::format_string<Ts...> fmt, Ts&&... ts);

template <typename... Ts>
void release_assert(bool condition, std::format_string<Ts...> fmt, Ts&&... ts);

template <typename... Ts>
void clean_assert(bool condition, std::format_string<Ts...> fmt, Ts&&... ts);

}  // namespace util

#include <inline/util/Asserts.inl>
