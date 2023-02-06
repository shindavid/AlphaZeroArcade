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
 * bool value = false;
 * options.make_options_description()
 *     ("on", store_bool(&value, true))
 *     ("off", store_bool(&value, false))
 *     ;
 *
 * https://stackoverflow.com/a/33172979/543913
 */
inline auto store_bool(bool* flag, bool store_as) {
  return boost::program_options::value(flag)->implicit_value(store_as)->zero_tokens()->default_value(*flag ^ store_as);
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
