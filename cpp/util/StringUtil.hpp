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
 */
template<int N=1024>
inline std::string create_string(char const* fmt, ...);

template<typename T>
void param_dump(const char* descr, const char* param_fmt, T param);

}  // namespace util

/*
 * PARAM_DUMP("abc", "%d", 3);
 * PARAM_DUMP("abcdef", "%.2f", 3.3);
 *
 * prints something like:
 *
 * abc:         3
 * abcdef:   3.30
 *
 * Note:
 * - The addition of colon and \n characters
 * - The right-alignment
 *
 * Additionally, this macro does some compile-time checking that the parameters look ok. Compilation will fail on
 * calls like:
 *
 * PARAM_DUMP("abc", "%f", 3);  // "%f" and 3 mismatch
 * PARAM_DUMP("abc", "d", 3);  // missing % in second arg
 * PARAM_DUMP("abc", "%3d", 3);  // digit shouldn't be here, alignment is the macro's responsibility
 */
#define PARAM_DUMP(descr, param_fmt, param) \
if (false) printf(param_fmt, param);        \
static_assert(param_fmt[0] == '%');         \
static_assert(param_fmt[1] != '-');         \
static_assert(param_fmt[1] != '+');         \
static_assert(!std::isdigit(param_fmt[1])); \
util::param_dump(descr, param_fmt, param);

#include <util/inl/StringUtil.inl>
