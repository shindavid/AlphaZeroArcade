#pragma once

/*
 * Various string utilities
 */
#include <util/Exception.hpp>

#include <cctype>
#include <cstdarg>
#include <iostream>
#include <string>
#include <vector>

namespace util {

inline std::string make_whitespace(size_t n) { return std::string(n, ' '); }

constexpr uint64_t str_hash(const char* c);

/*
 * Raises util::Exception if parse fails.
 */
float atof_safe(const std::string& s);

/*
 * split(s) and split(s, t) behave just like s.split() and s.split(t), respectively, in python.
 */
std::vector<std::string> split(const std::string& s, const char* t="");

/*
 * splitlines(s) behaves just like s.splitlines() in python.
 */
std::vector<std::string> splitlines(const std::string& s);

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

bool ends_with(const std::string& value, const std::string& ending);

}  // namespace util

#include <inline/util/StringUtil.inl>
