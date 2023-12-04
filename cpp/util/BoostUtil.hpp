#pragma once

#include <memory>
#include <string>

#include <boost/program_options.hpp>

#include <util/CppUtil.hpp>
#include <util/StringUtil.hpp>

namespace boost_util {

/*
 * get_option_value(util::split("--foo=bar --baz ..."), "foo") -> "bar"
 * get_option_value(util::split("--foo bar --baz ..."), "foo") -> "bar"
 *
 * If the option was not specified in args, returns the empty string.
 */
std::string get_option_value(const std::vector<std::string>& args, const std::string& option_name);

/*
 * Like get_option_value(), but also removes the option from args. Assumes that the given option is
 * a named arg, meaning that it is of the form "--foo=bar" or "--foo bar".
 */
std::string pop_option_value(std::vector<std::string>& args, const std::string& option_name);

namespace program_options {

/*
 * pos::value<float>(...)->default_value(...) sucks because it prints the default value with
 * undesirable precision. The official solution is to pass a second string argument to
 * default_value(), but that can be clunky. This float_value() function provides a cleaner way to
 * specify that string. Usage:
 *
 * boost_util::program_options::float_value("%.3f", &f)
 *
 * OR:
 *
 * boost_util::program_options::float_value("%.3f", &f, default_value)
 */
inline auto float_value(const char* fmt, float* dest, float default_value) {
  std::string s = util::create_string(fmt, default_value);
  return boost::program_options::value<float>(dest)->default_value(default_value, s);
}

inline auto float_value(const char* fmt, float* dest) { return float_value(fmt, dest, *dest); }

struct Settings {
  static bool help_full;
};

/*
 * This class is a thin wrapper around boost::program_options::options_description. It aims to
 * provide a similar interface, with the added benefit that option-naming clashes are detected at
 * compile-time, rather than at runtime.
 *
 * Before:
 *
 * using namespace po = boost::program_options;
 * po::options_description desc("descr");
 * desc.add_options()
 *     ("foo,f", ...)
 *     ("bar", ...)
 *     ;
 * return desc;
 *
 * After:
 *
 * using namespace po2 = boost_util::program_options;
 * po2::options_description desc("descr");
 * return desc
 *     .add_option<"foo", 'f'>(...)
 *     .add_option<"bar">(...)
 *     ;
 */
template <typename StrSeq_ = util::StringLiteralSequence<>,
          util::IntSequenceConcept CharSeq_ = std::integer_sequence<int>>
class options_description {
 public:
  using StrSeq = StrSeq_;
  using CharSeq = CharSeq_;

  using base_t = boost::program_options::options_description;

  options_description(const char* name) : full_base_(new base_t(name)), base_(new base_t(name)) {}
  ~options_description();

  /*
   * Similar to boost::program_options::options_description::add_options()(...), except that the
   * option name(s) is passed as a template argument, rather than as a function argument. This
   * allow for compile-time checking of name clashes.
   */
  template <util::StringLiteral StrLit, char Char = ' ', typename... Ts>
  auto add_option(Ts&&... ts);

  /*
   * Adds both --foo and --no-foo options. One of the two will be suppressed from the --help output,
   * depending on the value of *flag. This allows you to brainlessly add both options without
   * having to worry about what the default value is.
   *
   * Using --help-full instead of --help/-h will show both options.
   *
   * See: https://stackoverflow.com/a/33172979/543913
   */
  template <util::StringLiteral TrueStrLit, util::StringLiteral FalseStrLit>
  auto add_flag(bool* flag, const char* true_help, const char* false_help);

  /*
   * Adds all options from desc to this.
   */
  template <typename StrSeq2, util::IntSequenceConcept CharSeq2>
  auto add(const options_description<StrSeq2, CharSeq2>& desc);

  void print(std::ostream& s);

  const base_t& get() const { return *full_base_; }
  base_t& get() { return *full_base_; }

 private:
  options_description(base_t* full_base, base_t* base) : full_base_(full_base), base_(base) {}

  template <util::StringLiteral StrLit, char Char = ' '>
  auto augment() const;

  template <typename, util::IntSequenceConcept>
  friend class boost_util::program_options::options_description;

  base_t* full_base_;     // includes hidden options
  base_t* base_;          // excludes hidden options
  std::string tmp_str_;  // hack for convenience in augment() usage
};

/*
 * Constructs a boost::program_options::command_line_parser out of ts, which is expected to be
 * a collection of strings from the command line. Uses this to store to the passed-in desc, which
 * should be a {boost, boost_util}::program_options::options_description. Returns the parsed
 * variables_map.
 */
template <typename T, typename... Ts>
boost::program_options::variables_map parse_args(const T& desc, Ts&&... ts);

}  // namespace program_options

}  // namespace boost_util

#include <util/inl/BoostUtil.inl>
