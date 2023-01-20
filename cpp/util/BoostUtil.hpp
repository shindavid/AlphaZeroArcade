#pragma once

#include <string>

#include <boost/program_options.hpp>

#include <util/StringUtil.hpp>

namespace boost_util {

namespace program_options {

/*
 * pos::value<float>(...)->default_value(...) sucks because it prints the default value with undesirable
 * precision.
 *
 * boost_util::program_options::float_value("%.3f", &f)
 *
 * is a convenient helper that is equivalent to:
 *
 * boost::program_options::value<float>(&f)->default_value(f, util::create_string("%.3f", f))
 */
inline auto float_value(const char* fmt, float* dest, float default_value) {
  std::string s = util::create_string(fmt, default_value);
  return boost::program_options::value<float>(dest)->default_value(default_value, s);
}

inline auto float_value(const char* fmt, float* dest) { return float_value(fmt, dest, *dest); }

/*
 * abbrev_str(true, "abc", "a") == "abc,a"
 * abbrev_str(false, "abc", "a") == "abc"
 */
inline std::string abbrev_str(bool abbreviate, const char* full_name, const char* abbreviation) {
  return abbreviate ? util::create_string("%s,%s", full_name, abbreviation) : full_name;
}

}  // namespace program_options

}  // namespace boost_util
