#pragma once

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
auto float_value(const char* fmt, float* dest, float default_value) {
  std::string s = util::create_string(fmt, default_value);
  return boost::program_options::value<float>(dest)->default_value(default_value, s);
}

auto float_value(const char* fmt, float* dest) { return float_value(fmt, dest, *dest); }

}  // namespace program_options

}  // namespace boost_util
