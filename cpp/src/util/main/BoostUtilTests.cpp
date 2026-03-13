#include "util/BoostUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/Random.hpp"

#include <gtest/gtest.h>

#include <map>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

namespace compile_time_tests {

namespace po2 = boost_util::program_options;
using po2::options_description;

// Type aliases for descriptions with pre-populated options
using Desc0 = options_description<>;
using DescFoo = options_description<util::StringLiteralSequence<"foo">,
                                    std::integer_sequence<int, int('f')>>;
using DescBar = options_description<util::StringLiteralSequence<"bar">,
                                    std::integer_sequence<int, int('b')>>;

// SFINAE detection traits (workaround for GCC 13 bug with requires-expressions
// and constrained member function templates — see gcc.gnu.org/PR98644)
template <typename D, util::StringLiteral StrLit, char Char = ' ', typename = void>
struct can_add_option : std::false_type {};
template <typename D, util::StringLiteral StrLit, char Char>
struct can_add_option<D, StrLit, Char,
    std::void_t<decltype(std::declval<D&>().template add_option<StrLit, Char>(""))>>
    : std::true_type {};

template <typename D, util::StringLiteral StrLit, typename = void>
struct can_add_hidden_option : std::false_type {};
template <typename D, util::StringLiteral StrLit>
struct can_add_hidden_option<D, StrLit,
    std::void_t<decltype(std::declval<D&>().template add_hidden_option<StrLit>(""))>>
    : std::true_type {};

template <typename D, util::StringLiteral TrueLit, util::StringLiteral FalseLit,
          typename = void>
struct can_add_flag : std::false_type {};
template <typename D, util::StringLiteral TrueLit, util::StringLiteral FalseLit>
struct can_add_flag<D, TrueLit, FalseLit,
    std::void_t<decltype(std::declval<D&>().template add_flag<TrueLit, FalseLit>(
        (bool*)nullptr, "", ""))>>
    : std::true_type {};

template <typename D, util::StringLiteral TrueLit, util::StringLiteral FalseLit,
          typename = void>
struct can_add_hidden_flag : std::false_type {};
template <typename D, util::StringLiteral TrueLit, util::StringLiteral FalseLit>
struct can_add_hidden_flag<D, TrueLit, FalseLit,
    std::void_t<decltype(std::declval<D&>().template add_hidden_flag<TrueLit, FalseLit>(
        (bool*)nullptr, "", ""))>>
    : std::true_type {};

template <typename D, typename E, typename = void>
struct can_add_desc : std::false_type {};
template <typename D, typename E>
struct can_add_desc<D, E,
    std::void_t<decltype(std::declval<D&>().add(std::declval<const E&>()))>>
    : std::true_type {};

// -- add_option: valid cases --
static_assert(can_add_option<Desc0, "foo", 'f'>::value);
static_assert(can_add_option<DescFoo, "bar", 'b'>::value);
static_assert(can_add_option<DescFoo, "bar">::value);  // no abbreviation

// -- add_option: name clash --
static_assert(!can_add_option<DescFoo, "foo">::value);
static_assert(!can_add_option<DescFoo, "foo", 'x'>::value);

// -- add_option: abbreviation clash --
static_assert(!can_add_option<DescFoo, "bar", 'f'>::value);

// -- add_hidden_option: valid --
static_assert(can_add_hidden_option<Desc0, "foo">::value);
static_assert(can_add_hidden_option<DescFoo, "baz">::value);

// -- add_hidden_option: name clash --
static_assert(!can_add_hidden_option<DescFoo, "foo">::value);

// -- add_flag: valid --
static_assert(can_add_flag<Desc0, "enable", "disable">::value);
static_assert(can_add_flag<DescFoo, "enable", "disable">::value);

// -- add_flag: true name clashes with existing --
static_assert(!can_add_flag<DescFoo, "foo", "bar">::value);

// -- add_flag: false name clashes with existing --
static_assert(!can_add_flag<DescFoo, "bar", "foo">::value);

// -- add_flag: true == false --
static_assert(!can_add_flag<Desc0, "same", "same">::value);

// -- add_hidden_flag: valid --
static_assert(can_add_hidden_flag<Desc0, "yes", "no">::value);

// -- add_hidden_flag: name clash --
static_assert(!can_add_hidden_flag<DescFoo, "foo", "bar">::value);

// -- add: valid merge of non-overlapping descriptions --
static_assert(can_add_desc<DescFoo, DescBar>::value);

// -- add: name clash on merge --
static_assert(!can_add_desc<DescFoo, DescFoo>::value);

// -- add: abbreviation clash on merge --
using DescBazF = options_description<util::StringLiteralSequence<"baz">,
                                     std::integer_sequence<int, int('f')>>;
static_assert(!can_add_desc<DescFoo, DescBazF>::value);

}  // namespace compile_time_tests

TEST(BoostUtil, compile_time_option_clash_detection) {
  // All actual checking is done via static_assert above at compile time.
  // This test exists to document that the checks are present and to provide
  // a visible test entry point.
  SUCCEED();
}

TEST(BoostUtil, get_random_set_index) {
  util::Random::set_seed(1);

  int n = 10;
  boost::dynamic_bitset<> bitset(n);
  int index;

  index = boost_util::get_random_set_index(bitset);
  EXPECT_EQ(index, -1);

  bitset.set(1);
  bitset.set(3);
  bitset.set(5);
  bitset.set(7);
  bitset.set(9);

  int N = 1000;
  using count_map_t = std::map<int, int>;
  count_map_t counts;

  for (int i = 0; i < N; ++i) {
    index = boost_util::get_random_set_index(bitset);
    counts[index]++;
    EXPECT_LT(index, n);
    EXPECT_GE(index, 0);
    EXPECT_TRUE(bitset.test(index));
  }

  int count = bitset.count();
  int c = bitset.find_first();
  int i = count;
  while (i-- > 0) {
    EXPECT_GT(counts[c], N / (2 * count));
    c = bitset.find_next(c);
  }
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
