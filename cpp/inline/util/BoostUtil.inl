#include "util/BoostUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/ScreenUtil.hpp"

#include <format>
#include <fstream>

namespace boost_util {

namespace program_options {

template <typename StrSeq, util::concepts::IntSequence CharSeq>
options_description<StrSeq, CharSeq>::options_description(const char* name)
    : full_base_(new base_t(name, util::get_screen_width() - 1)),
      base_(new base_t(name, util::get_screen_width() - 1)) {}

template <typename StrSeq, util::concepts::IntSequence CharSeq>
options_description<StrSeq, CharSeq>::~options_description() {
  // This is bad but we deliberately leak the base_t objects here. Deleting them leads to a
  // segfault due to how boost uses these objects. There is probably a workaround but I could not
  // figure it out.
}

template <typename StrSeq, util::concepts::IntSequence CharSeq>
template <util::StringLiteral StrLit, char Char, typename... Ts>
auto options_description<StrSeq, CharSeq>::add_option(Ts&&... ts) {
  auto out = augment<StrLit, Char>();
  std::string full_name = out.tmp_str_;

  out.full_base_->add_options()(full_name.c_str(), std::forward<Ts>(ts)...);
  out.base_->add_options()(full_name.c_str(), std::forward<Ts>(ts)...);
  return out;
}

template <typename StrSeq, util::concepts::IntSequence CharSeq>
template <util::StringLiteral StrLit, typename... Ts>
auto options_description<StrSeq, CharSeq>::add_hidden_option(Ts&&... ts) {
  auto out = augment<StrLit>();
  std::string full_name = out.tmp_str_;

  out.full_base_->add_options()(full_name.c_str(), std::forward<Ts>(ts)...);
  return out;
}

template <typename StrSeq, util::concepts::IntSequence CharSeq>
template <util::StringLiteral TrueStrLit, util::StringLiteral FalseStrLit>
auto options_description<StrSeq, CharSeq>::add_flag(bool* flag, const char* true_help,
                                                    const char* false_help) {
  return add_flag_helper<TrueStrLit, FalseStrLit>(flag, true_help, false_help, false);
}

template <typename StrSeq, util::concepts::IntSequence CharSeq>
template <util::StringLiteral TrueStrLit, util::StringLiteral FalseStrLit>
auto options_description<StrSeq, CharSeq>::add_hidden_flag(bool* flag, const char* true_help,
                                                           const char* false_help) {
  return add_flag_helper<TrueStrLit, FalseStrLit>(flag, true_help, false_help, true);
}

template <typename StrSeq, util::concepts::IntSequence CharSeq>
template <typename StrSeq2, util::concepts::IntSequence CharSeq2>
auto options_description<StrSeq, CharSeq>::add(const options_description<StrSeq2, CharSeq2>& desc) {
  static_assert(util::no_overlap_v<StrSeq, StrSeq2>, "Options name clash!");
  static_assert(util::no_overlap_v<CharSeq, CharSeq2>, "Options abbreviation clash!");

  using StrSeq3 = util::concat_string_literal_sequence_t<StrSeq, StrSeq2>;
  using CharSeq3 = util::concat_int_sequence_t<CharSeq, CharSeq2>;
  using OutT = options_description<StrSeq3, CharSeq3>;

  full_base_->add(*desc.full_base_);
  base_->add(*desc.base_);

  OutT out(full_base_, base_);
  return out;
}

template <typename StrSeq, util::concepts::IntSequence CharSeq>
void options_description<StrSeq, CharSeq>::print(std::ostream& s) const {
  if (Settings::help_full) {
    full_base_->print(s);
  } else {
    base_->print(s);
  }
}

template <typename StrSeq, util::concepts::IntSequence CharSeq>
template <util::StringLiteral StrLit, char Char>
auto options_description<StrSeq, CharSeq>::augment() const {
  static_assert(!util::string_literal_sequence_contains_v<StrSeq, StrLit>, "Options name clash!");
  constexpr bool UsingAbbrev = Char != ' ';
  static_assert(!UsingAbbrev || !util::int_sequence_contains_v<CharSeq, int(Char)>,
                "Options abbreviation clash!");

  using StrSeq2 =
    util::concat_string_literal_sequence_t<StrSeq, util::StringLiteralSequence<StrLit>>;
  using CharSeq2 = std::conditional_t<
    UsingAbbrev, util::concat_int_sequence_t<CharSeq, util::int_sequence<int(Char)>>, CharSeq>;
  using OutT = options_description<StrSeq2, CharSeq2>;

  OutT out(full_base_, base_);

  std::string full_name(StrLit.value);
  if (UsingAbbrev) {
    full_name = std::format("{},{}", full_name, Char);
  }
  out.tmp_str_ = full_name;

  return out;
}

template <typename StrSeq, util::concepts::IntSequence CharSeq>
template <util::StringLiteral TrueStrLit, util::StringLiteral FalseStrLit>
auto options_description<StrSeq, CharSeq>::add_flag_helper(bool* flag, const char* true_help,
                                                           const char* false_help, bool hidden) {
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

  out.full_base_->add_options()(true_name, po::value(flag)->implicit_value(true)->zero_tokens(),
                                full_true_help.c_str())(
    false_name, po::value(flag)->implicit_value(false)->zero_tokens(), full_false_help.c_str());

  if (!hidden) {
    if (*flag) {
      out.base_->add_options()(false_name, po::value(flag)->implicit_value(false)->zero_tokens(),
                               full_false_help.c_str());
    } else {
      out.base_->add_options()(true_name, po::value(flag)->implicit_value(true)->zero_tokens(),
                               full_true_help.c_str());
    }
  }

  return out;
}

namespace detail {

template <typename T>
struct Wrap {
  const T& operator()(const T& t) const { return t; }
};

template <typename S, util::concepts::IntSequence C>
struct Wrap<options_description<S, C>> {
  const auto& operator()(const options_description<S, C>& t) const { return t.get(); }
};

template <typename T>
const auto& wrap(const T& t) {
  return Wrap<T>()(t);
}

}  // namespace detail

template <typename T, typename... Ts>
boost::program_options::variables_map parse_args(const T& desc, Ts&&... ts) {
  namespace po = boost::program_options;
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(std::forward<Ts>(ts)...).options(detail::wrap(desc)).run(),
              vm);
  } catch (const po::error& e) {
    throw util::CleanException("{}", e.what());
  }
  po::notify(vm);
  return vm;
}

}  // namespace program_options

inline void write_str_to_file(const std::string& str, const boost::filesystem::path& filename) {
  std::ofstream file(filename);  // Convert path to string for ofstream
  if (file.is_open()) {
    file << str;
    file.close();
  } else {
    throw std::runtime_error("Unable to open file: " + filename.string());
  }
}

}  // namespace boost_util
