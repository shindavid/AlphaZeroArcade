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

/*
 * bool value = false;
 * options.make_options_description()
 *     ("on", store_bool(&value, true))
 *     ("off", store_bool(&value, false))
 *     ;
 *
 * https://stackoverflow.com/a/33172979/543913
 */
inline auto store_bool(bool* flag, bool store_as) {
  return boost::program_options::value(flag)->implicit_value(store_as)->zero_tokens()->default_value(*flag ^ !store_as);
}

/*
 * The above store_bool() mechanism is not compatible with ->default_value() for whatever reason.
 *
 * This helper function is a convenience that sticks "... (default: true)" or "... (default: false)" at the end of
 * help string.
 */
inline std::string make_store_bool_help_str(const char* help, bool default_value) {
  return util::create_string("%s (default: %s)", help, default_value ? "true" : "false");
}

}  // namespace program_options

}  // namespace boost_util
