#pragma once

#include "util/CppUtil.hpp"

#include <boost/dynamic_bitset.hpp>
#include <boost/filesystem.hpp>
#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <format>
#include <random>
#include <string>

namespace boost_util {

// Returns a random index of a set bit in the given bitset. If no bits are set, returns -1.
//
// Uses prng as the random number generator.
int get_random_set_index(std::mt19937& prng, const boost::dynamic_bitset<>& bitset);

// Returns a random index of a set bit in the given bitset. If no bits are set, returns -1.
//
// Uses util::Random::default_prng() as the random number generator.
int get_random_set_index(const boost::dynamic_bitset<>& bitset);

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

void pretty_print(std::ostream& os, boost::json::value const& jv, std::string* indent = nullptr);

void write_str_to_file(const std::string& str, const boost::filesystem::path& filename);

namespace program_options {

/*
 * pos::value<float>(...)->default_value(...) sucks because it prints the default value with
 * undesirable precision. The official solution is to pass a second string argument to
 * default_value(), but that can be clunky. This default_value() function provides a cleaner way to
 * specify that string. Usage:
 *
 * boost_util::program_options::default_value("{:.3f}", &f)
 *
 * OR:
 *
 * boost_util::program_options::default_value("{:.3f}", &f, default_value)
 */
template <typename T>
auto default_value(std::format_string<T> fmt, T* dest, T t) {
  // move t into format so it's an rvalue and deduces Args... = {T}
  std::string s = std::format(fmt, std::move(t));
  return boost::program_options::value<T>(dest)->default_value(t, s);
}

template <typename T>
auto default_value(std::format_string<T> fmt, T* dest) {
  return default_value(fmt, dest, *dest);
}

struct Settings {
  static inline bool help_full = false;
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
          util::concepts::IntSequence CharSeq_ = std::integer_sequence<int>>
class options_description {
 public:
  using StrSeq = StrSeq_;
  using CharSeq = CharSeq_;

  using base_t = boost::program_options::options_description;

  options_description(const char* name);
  ~options_description();

  /*
   * Similar to boost::program_options::options_description::add_options()(...), except that the
   * option name(s) is passed as a template argument, rather than as a function argument. This
   * allow for compile-time checking of name clashes.
   */
  template <util::StringLiteral StrLit, char Char = ' ', typename... Ts>
  auto add_option(Ts&&... ts);

  /*
   * Similar to add_option(), but keeps the option hidden from the --help output to avoid clutter.
   *
   * To actually see the hidden options, use --help-full instead of --help/-h.
   *
   * For hidden options, we don't allow single-character abbreviations.
   */
  template <util::StringLiteral StrLit, typename... Ts>
  auto add_hidden_option(Ts&&... ts);

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
   * Like add_flag(), but both options are hidden from the --help output.
   */
  template <util::StringLiteral TrueStrLit, util::StringLiteral FalseStrLit>
  auto add_hidden_flag(bool* flag, const char* true_help, const char* false_help);

  /*
   * Adds all options from desc to this.
   */
  template <typename StrSeq2, util::concepts::IntSequence CharSeq2>
  auto add(const options_description<StrSeq2, CharSeq2>& desc);

  void print(std::ostream& s) const;

  friend std::ostream& operator<<(std::ostream& s, const options_description& desc) {
    desc.print(s);
    return s;
  }

  const base_t& get() const { return *full_base_; }
  base_t& get() { return *full_base_; }

 private:
  options_description(base_t* full_base, base_t* base) : full_base_(full_base), base_(base) {}

  template <util::StringLiteral StrLit, char Char = ' '>
  auto augment() const;

  template <util::StringLiteral TrueStrLit, util::StringLiteral FalseStrLit>
  auto add_flag_helper(bool* flag, const char* true_help, const char* false_help, bool hidden);

  template <typename, util::concepts::IntSequence>
  friend class boost_util::program_options::options_description;

  base_t* full_base_;    // includes hidden options
  base_t* base_;         // excludes hidden options
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

#include "inline/util/BoostUtil.inl"
