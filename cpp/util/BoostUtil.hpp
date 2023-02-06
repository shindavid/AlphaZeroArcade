#pragma once

#include <string>

#include <boost/program_options.hpp>

#include <util/CppUtil.hpp>
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
 * This class is a thin wrapper around boost::program_options::options_description. It aims to provide a similar
 * interface, with the added benefit that option-naming clashes are detected at compile-time, rather than at
 * runtime.
 *
 * Before:
 *
 * using namespace po = boost::program_options;
 * po::options_description desc("descr");
 * desc.add_options()
 *     ("foo,f", ...)
 *     ("bar,b", ...)
 *     ;
 * return desc;
 *
 * After:
 *
 * using namespace po2 = boost_util::program_options;
 * po2::options_description desc("descr");
 * return desc
 *     .add_option("foo", 'f', ...)
 *     .add_option("bar", 'b', ...)
 *     ;
 */
template<
    typename StringLiteralSequence_ = util::StringLiteralSequence<>,
    util::IntSequenceConcept CharSeq_=std::integer_sequence<int>>
class options_description : public boost::program_options::options_description
{
public:
  using StringLiteralSequence = StringLiteralSequence_;
  using CharSeq = CharSeq_;

  using base_t = boost::program_options::options_description;

  template<typename... Ts>
  options_description(Ts&&... ts) : base_t(std::forward<Ts>(ts)...) {}

  template<util::StringLiteral StrLit, char Char=' ', typename... Ts>
  auto add_option(Ts&&... ts) {
    static_assert(!util::string_literal_sequence_contains_v<StringLiteralSequence, StrLit>, "Options name clash!");
    constexpr bool UsingAbbrev = Char != ' ';
    static_assert(!UsingAbbrev || !util::int_sequence_contains_v<CharSeq, int(Char)>, "Options abbreviation clash!");

    using StringLiteralSequence2 = util::concat_string_literal_sequence_t<
        StringLiteralSequence, util::StringLiteralSequence<StrLit>>;
    using CharSeq2 = std::conditional_t<
        UsingAbbrev, util::concat_int_sequence_t<CharSeq, util::int_sequence<int(Char)>>, CharSeq>;
    using OutT = options_description<StringLiteralSequence2, CharSeq2>;

    OutT out(*this);
    std::string full_name(StrLit.value);
    if (UsingAbbrev) {
      full_name = util::create_string("%s,%c", full_name.c_str(), Char);
    }

    out.add_options()(full_name.c_str(), std::forward<Ts>(ts)...);
    return out;
  }

  /*
   * Adds both --foo and --no-foo options.
   *
   * See: https://stackoverflow.com/a/33172979/543913
   */
  template<util::StringLiteral TrueStrLit, util::StringLiteral FalseStrLit>
  auto add_bool_switches(bool* flag, const char* true_help, const char* false_help) {
    namespace po = boost::program_options;

    std::string full_true_help = (*flag) ? "no-op" : true_help;
    std::string full_false_help = (*flag) ? false_help : "no-op";

    return (*this)
        .template add_option<TrueStrLit>(po::value(flag)->implicit_value(true)->zero_tokens(), full_true_help.c_str())
        .template add_option<FalseStrLit>(po::value(flag)->implicit_value(false)->zero_tokens(), full_false_help.c_str())
        ;
  }

  template<
      typename StringLiteralSequence2,
      util::IntSequenceConcept CharSeq2>
  auto add(const options_description<StringLiteralSequence2, CharSeq2>& desc) {
    static_assert(util::no_overlap_v<StringLiteralSequence, StringLiteralSequence2>, "Options name clash!");
    static_assert(util::no_overlap_v<CharSeq, CharSeq2>, "Options abbreviation clash!");

    using StringLiteralSequence3 = util::concat_string_literal_sequence_t<
        StringLiteralSequence, StringLiteralSequence2>;
    using CharSeq3 = util::concat_int_sequence_t<CharSeq, CharSeq2>;
    using OutT = options_description<StringLiteralSequence3, CharSeq3>;

    OutT out(*this);
    out.add_wrapper(desc);
    return out;
  }

private:
  void add_wrapper(const boost::program_options::options_description& desc) {
    base_t::add(desc);
  }

  template<typename, util::IntSequenceConcept> friend class boost_util::program_options::options_description;
};

}  // namespace program_options

}  // namespace boost_util
