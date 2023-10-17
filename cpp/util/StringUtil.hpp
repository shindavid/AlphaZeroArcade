#pragma once

/*
 * Various string utilities
 */

#include <cctype>
#include <cstdarg>
#include <iostream>
#include <string>
#include <vector>

#include <util/Exception.hpp>

namespace util {

constexpr uint64_t str_hash(const char* c);

/*
 * Raises util::Exception if parse fails.
 */
float atof_safe(const std::string& s);

/*
 * split(s) behaves just like s.split() in python.
 *
 * TODO: support split(s, t), to behave like s.split(t).
 */
std::vector<std::string> split(const std::string& s);

/*
 * Like sprintf(), but conveniently bypasses need to declare the char buffer at the call-site.
 *
 * The template parameter N dictates the char buffer size. In case of overflow, throws an exception.
 *
 * TODO: once the std::format library is implemented in gcc, use that instead of printf-style
 * formatting.
 */
template <int N = 1024>
inline std::string create_string(char const* fmt, ...);

}  // namespace util

#include <util/inl/StringUtil.inl>
