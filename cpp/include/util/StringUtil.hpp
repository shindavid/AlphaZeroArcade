#pragma once

/*
 * Various string utilities
 */
#include <cctype>
#include <cstdarg>
#include <cstdint>
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
std::vector<std::string> split(const std::string& s, const char* t = "");

// Similar to split(s, t), with some notable differences:
//
// - Writes the tokens to result instead of returning a new vector.
// - If the passed-in result is longer than the number of tokens, the excess entries in result are
//   left unchanged.
// - Returns the number of tokens found, which may be less than the size of result.
//
// The benefit of this function over split(s, t) is that it is more efficient when called
// repeatedly with the same result vector. If the s/t arguments are assumed to be drawn iid from
// some distribution, then the expected number of dynamic memory allocations will asymptotically
// approach zero.
int split(std::vector<std::string>& result, const std::string& s, const char* t = "");

/*
 * splitlines(s) behaves just like s.splitlines() in python.
 */
std::vector<std::string> splitlines(const std::string& s);

bool ends_with(const std::string& value, const std::string& ending);

/*
 * Returns the width of the string when printed to the terminal. This is essentially the number of
 * characters in the string, except that control characters are treated as having zero width.
 */
size_t terminal_width(const std::string& str);

// "x and y"
// "x, y, and z"  (oxford_comma = true)
// "x, y and z" (oxford_comma = false)
std::string grammatically_join(const std::vector<std::string>& items,
                               const std::string& conjunction, bool oxford_comma = true);

uint64_t parse_bytes(const std::string& str);  // supports 256MiB / 256MB / 1GiB / 1073741824

}  // namespace util

#include "inline/util/StringUtil.inl"
