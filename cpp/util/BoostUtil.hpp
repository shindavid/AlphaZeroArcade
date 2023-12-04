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

struct Settings {
  static bool help_full;
};

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
template <typename StringLiteralSequence_ = util::StringLiteralSequence<>,
          util::IntSequenceConcept CharSeq_ = std::integer_sequence<int>>
class options_description {
 public:
  using StringLiteralSequence = StringLiteralSequence_;
  using CharSeq = CharSeq_;

  using base_t = boost::program_options::options_description;

  options_description(base_t* full_base, base_t* base) : full_base_(full_base), base_(base) {}

  options_description(const char* name) : full_base_(new base_t(name)), base_(new base_t(name)) {}

  // template <typename... Ts>
  // options_description(Ts&&... ts)
  //     : full_base_(new base_t(std::forward<Ts>(ts)...)),
  //       base_(new base_t(std::forward<Ts>(ts)...)) {}

  ~options_description() {
    // This is bad but we deliberately leak the base_t objects here. Deleting them leads to a
    // segfault due to how boost uses these objects. There is probably a workaround but I could not
    // figure it out.
  }

  template <util::StringLiteral StrLit, char Char = ' ', typename... Ts>
  auto add_option(Ts&&... ts) {
    auto out = augment<StrLit, Char>();
    std::string full_name = out.tmp_str_;

    out.full_base_->add_options()(full_name.c_str(), std::forward<Ts>(ts)...);
    out.base_->add_options()(full_name.c_str(), std::forward<Ts>(ts)...);
    return out;
  }

  /*
   * Adds both --foo and --no-foo options. One of the two will be suppressed from the --help output,
   * depending on the value of *flag. This can be useful for boolean flags where you anticipate that
   * you might want to change the default value, either directly in the code or via some sort of
   * runtime switch.
   *
   * See: https://stackoverflow.com/a/33172979/543913
   */
  template <util::StringLiteral TrueStrLit, util::StringLiteral FalseStrLit>
  auto add_bool_switches(bool* flag, const char* true_help, const char* false_help) {
    auto out = augment<TrueStrLit>().template augment<FalseStrLit>();

    std::string full_true_help = true_help;
    std::string full_false_help = false_help;

    if (*flag) {
      full_true_help += " (no-op)";
    } else {
      full_false_help += " (no-op)";
    }

    const char* true_name = TrueStrLit.value;
    const char* false_name = FalseStrLit.value;

    namespace po = boost::program_options;

    out.full_base_->add_options()
    (true_name, po::value(flag)->implicit_value(true)->zero_tokens(), full_true_help.c_str())
    (false_name, po::value(flag)->implicit_value(false)->zero_tokens(), full_false_help.c_str());

    if (*flag) {
      out.base_->add_options()(false_name, po::value(flag)->implicit_value(false)->zero_tokens(),
                              full_false_help.c_str());
    } else {
      out.base_->add_options()(true_name, po::value(flag)->implicit_value(true)->zero_tokens(),
                              full_true_help.c_str());
    }

    return out;
  }

  template <typename StringLiteralSequence2, util::IntSequenceConcept CharSeq2>
  auto add(const options_description<StringLiteralSequence2, CharSeq2>& desc) {
    static_assert(util::no_overlap_v<StringLiteralSequence, StringLiteralSequence2>,
                  "Options name clash!");
    static_assert(util::no_overlap_v<CharSeq, CharSeq2>, "Options abbreviation clash!");

    using StringLiteralSequence3 =
        util::concat_string_literal_sequence_t<StringLiteralSequence, StringLiteralSequence2>;
    using CharSeq3 = util::concat_int_sequence_t<CharSeq, CharSeq2>;
    using OutT = options_description<StringLiteralSequence3, CharSeq3>;

    full_base_->add(*desc.full_base_);
    base_->add(*desc.base_);

    OutT out(full_base_, base_);
    return out;
  }

  void print(std::ostream& s) {
    if (Settings::help_full) {
      full_base_->print(s);
    } else {
      base_->print(s);
    }
  }

  const base_t& get() const { return *full_base_; }
  base_t& get() { return *full_base_; }

 private:
  template <util::StringLiteral StrLit, char Char=' '>
  auto augment() const {
    static_assert(!util::string_literal_sequence_contains_v<StringLiteralSequence, StrLit>,
                  "Options name clash!");
    constexpr bool UsingAbbrev = Char != ' ';
    static_assert(!UsingAbbrev || !util::int_sequence_contains_v<CharSeq, int(Char)>,
                  "Options abbreviation clash!");

    using StringLiteralSequence2 =
        util::concat_string_literal_sequence_t<StringLiteralSequence,
                                               util::StringLiteralSequence<StrLit>>;
    using CharSeq2 = std::conditional_t<
        UsingAbbrev, util::concat_int_sequence_t<CharSeq, util::int_sequence<int(Char)>>, CharSeq>;
    using OutT = options_description<StringLiteralSequence2, CharSeq2>;

    OutT out(full_base_, base_);

    std::string full_name(StrLit.value);
    if (UsingAbbrev) {
      full_name = util::create_string("%s,%c", full_name.c_str(), Char);
    }
    out.tmp_str_ = full_name;

    return out;
  }

  template <typename, util::IntSequenceConcept>
  friend class boost_util::program_options::options_description;

  base_t* full_base_;     // includes hidden options
  base_t* base_;          // excludes hidden options
  std::string tmp_str_;  // hack for convenience in augment() usage
};

template<typename T>
struct _Wrap {
  const T& operator()(const T& t) const { return t; }
};

template <typename S, util::IntSequenceConcept C>
struct _Wrap<options_description<S, C>> {
  const auto& operator()(const options_description<S, C>& t) const {
    return t.get();
  }
};

template<typename T>
const auto& wrap(const T& t) { return _Wrap<T>()(t); }

template <typename T, typename... Ts>
boost::program_options::variables_map parse_args(const T& desc, Ts&&... ts) {
  namespace po = boost::program_options;
  po::variables_map vm;
  po::store(po::command_line_parser(std::forward<Ts>(ts)...).options(wrap(desc)).run(), vm);
  po::notify(vm);
  return vm;
}

}  // namespace program_options

}  // namespace boost_util

#include <util/inl/BoostUtil.inl>
