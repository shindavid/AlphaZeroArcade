#pragma once

/*
 * Various string utilities
 */

#include <cstdarg>
#include <string>
#include <vector>

#include <util/Exception.hpp>

namespace util {

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

}  // namespace util

#include <util/StringUtilINLINES.cpp>
