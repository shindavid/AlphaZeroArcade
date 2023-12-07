#pragma once

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

void debug_assert(bool condition, char const* fmt = nullptr, ...)
    __attribute__((format(printf, 2, 3)));
void release_assert(bool condition, char const* fmt = nullptr, ...)
    __attribute__((format(printf, 2, 3)));
void clean_assert(bool condition, char const* fmt = nullptr, ...)
    __attribute__((format(printf, 2, 3)));

}  // namespace util

#include <util/inl/Asserts.inl>
