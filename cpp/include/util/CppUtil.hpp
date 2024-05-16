#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <tuple>
#include <typeinfo>
#include <type_traits>
#include <unistd.h>

#include <boost/core/demangle.hpp>

#define CONCAT_HELPER(x, y) x##y
#define CONCAT(x, y) CONCAT_HELPER(x, y)

#define XSTR(a) STR(a)
#define STR(a) #a

/*
 * Useful macro for constexpr-detection of whether a macro is assigned to 1. This is useful given
 * the behavior of the -D option in py/build.py.
 *
 * #define FOO 1
 * // #define BAR
 *
 * static_assert(IS_MACRO_ENABLED(FOO))
 * static_assert(!IS_MACRO_ENABLED(BAR))
 */
#define IS_MACRO_ENABLED(macro) (XSTR(macro)[0] == '1')

namespace util {

template <typename T>
std::string get_typename() {
  return boost::core::demangle(typeid(T).name());
}
template <typename T>
std::string get_typename(const T& t) {
  return get_typename<T>();
}

/*
 * Returns true if the output is a terminal, false otherwise.
 *
 * This is useful for determining whether to print color codes.
 *
 * https://stackoverflow.com/a/5157076/543913
 */
inline bool tty_mode() { return isatty(STDOUT_FILENO); }

int64_t constexpr inline s_to_ns(int64_t s) { return s * 1000 * 1000 * 1000; }
int64_t constexpr inline us_to_ns(int64_t us) { return us * 1000; }
int64_t constexpr inline ms_to_ns(int64_t ms) { return ms * 1000 * 1000; }

/*
 * Usage:
 *
 * int64_t ns = util::ns_since_epoch(std::chrono::steady_clock::now());
 */
template <typename TimePoint>
int64_t ns_since_epoch(const TimePoint&);

inline int64_t ns_since_epoch() { return ns_since_epoch(std::chrono::system_clock::now()); }

/*
 * Between machine reboots, no two calls to this function from the same machine should return equal
 * values.
 */
int64_t get_unique_id();

/*
 * This identity function is intended to be used to declare required members in concepts.
 *
 * Example usage:
 *
 * template <class T>
 * concept Foo = requires(T t) {
 *   { util::decay_copy(T::bar) } -> std::is_same_as<int>;
 * };
 *
 * The above expresses the requirement that the class T has a static member bar of type int.
 *
 * This function does not actually have an implementation, and so you will get a linker error if
 * you try to actually invoke it in other contexts.
 *
 * Adapted from: https://stackoverflow.com/a/69687663/543913
 */
template <class T>
std::decay_t<T> decay_copy(T&&);

template <size_t N>
struct StringLiteral {
  constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }
  template <size_t M>
  constexpr bool operator==(const StringLiteral<M>& other) const {
    // Caveman-style implementation of strcmp(). Doing this because c++ standard does not require
    // strcmp() to be constexpr. Clang in particular does not mark strcmp() as constexpr, and clang
    // powers CLion.
    if (N != M) return false;
    for (size_t i = 0; i < N; ++i) {
      if (value[i] != other.value[i]) return false;
    }
    return true;
  }
  char value[N];
};

template <StringLiteral...>
struct StringLiteralSequence {};

/*
 * The following are equivalent:
 *
 * using T = util::int_sequence<1, 2, 3>;
 *
 * and:
 *
 * using T = std::integer_sequence<int, 1, 2, 3>;
 */
template <int... Ints>
using int_sequence = std::integer_sequence<int, Ints...>;
template <int64_t... Ints>
using int64_sequence = std::integer_sequence<int64_t, Ints...>;
template <uint64_t... Ints>
using uint64_sequence = std::integer_sequence<uint64_t, Ints...>;

/*
 * IntSequenceConcept/Int64SequenceConcept/UInt64SequenceConcept is for concept requirements.
 */
template <typename T>
struct is_int_sequence {
  static const bool value = false;
};
template <int... Ints>
struct is_int_sequence<int_sequence<Ints...>> {
  static constexpr bool value = true;
};
template <typename T>
inline constexpr bool is_int_sequence_v = is_int_sequence<T>::value;
template <typename T>
concept IntSequenceConcept = is_int_sequence_v<T>;

template <typename T>
struct is_int64_sequence {
  static const bool value = false;
};
template <int64_t... Ints>
struct is_int64_sequence<int64_sequence<Ints...>> {
  static constexpr bool value = true;
};
template <typename T>
inline constexpr bool is_int64_sequence_v = is_int64_sequence<T>::value;
template <typename T>
concept Int64SequenceConcept = is_int64_sequence_v<T>;

template <typename T>
struct is_uint64_sequence {
  static const bool value = false;
};
template <uint64_t... Ints>
struct is_uint64_sequence<uint64_sequence<Ints...>> {
  static constexpr bool value = true;
};
template <typename T>
inline constexpr bool is_uint64_sequence_v = is_uint64_sequence<T>::value;
template <typename T>
concept UInt64SequenceConcept = is_uint64_sequence_v<T>;

template <typename T>
struct integer_sequence_product {};
template <typename T>
struct integer_sequence_product<std::integer_sequence<T>> {
  static constexpr T value = 1;
};
template <typename T, T I, T... Is>
struct integer_sequence_product<std::integer_sequence<T, I, Is...>> {
  static constexpr T value = I * integer_sequence_product<std::integer_sequence<T, Is...>>::value;
};
template <typename T>
static constexpr auto integer_sequence_product_v = integer_sequence_product<T>::value;

template <typename T, T... values>
constexpr T get_value(const std::integer_sequence<T, values...>&, const size_t idx) {
  constexpr T vals[] = {values...};
  return vals[idx];
}

/*
 * true: util::int_sequence_contains_v<util::int_sequence<1, 3, 5>, 1>
 * true: util::int_sequence_contains_v<util::int_sequence<1, 3, 5>, 5>
 * false: util::int_sequence_contains_v<util::int_sequence<1, 3, 5>, 2>
 *
 * Similarly we have int64_sequence_contains_v and uint64_sequence_contains_v.
 */
template <typename T, int K>
struct int_sequence_contains {
  static constexpr bool value = false;
};
template <int I, int... Is, int K>
struct int_sequence_contains<int_sequence<I, Is...>, K> {
  static constexpr bool value = (I == K) || int_sequence_contains<int_sequence<Is...>, K>::value;
};
template <typename T, int K>
static constexpr bool int_sequence_contains_v = int_sequence_contains<T, K>::value;

template <typename T, int64_t K>
struct int64_sequence_contains {
  static constexpr bool value = false;
};
template <int64_t I, int64_t... Is, int64_t K>
struct int64_sequence_contains<int64_sequence<I, Is...>, K> {
  static constexpr bool value =
      (I == K) || int64_sequence_contains<int64_sequence<Is...>, K>::value;
};
template <typename T, int64_t K>
static constexpr bool int64_sequence_contains_v = int64_sequence_contains<T, K>::value;

template <typename T, uint64_t K>
struct uint64_sequence_contains {
  static constexpr bool value = false;
};
template <uint64_t I, uint64_t... Is, uint64_t K>
struct uint64_sequence_contains<uint64_sequence<I, Is...>, K> {
  static constexpr bool value =
      (I == K) || uint64_sequence_contains<uint64_sequence<Is...>, K>::value;
};
template <typename T, uint64_t K>
static constexpr bool uint64_sequence_contains_v = uint64_sequence_contains<T, K>::value;

template <typename T, StringLiteral S>
struct string_literal_sequence_contains {
  static constexpr bool value = false;
};
template <StringLiteral I, StringLiteral... Is, StringLiteral S>
struct string_literal_sequence_contains<StringLiteralSequence<I, Is...>, S> {
  static constexpr bool value =
      (I == S) || string_literal_sequence_contains<StringLiteralSequence<Is...>, S>::value;
};
template <typename T, StringLiteral S>
static constexpr bool string_literal_sequence_contains_v =
    string_literal_sequence_contains<T, S>::value;

/*
 * BitSetConcept<T> is for concept requirements.
 */
template <typename T>
struct is_bit_set {
  static const bool value = false;
};
template <size_t N>
struct is_bit_set<std::bitset<N>> {
  static const bool value = true;
};
template <typename T>
inline constexpr bool is_bit_set_v = is_bit_set<T>::value;
template <typename T>
concept BitSetConcept = is_bit_set_v<T>;

/*
 * The following are equivalent:
 *
 * using S = util::int_sequence<1, 2>;
 * using T = util::int_sequence<3>;
 * using U = util::concat_int_sequence_t<S, T>;
 *
 * and:
 *
 * using U = util::int_sequence<1, 2, 3>;
 */
template <typename T, typename U>
struct concat_int_sequence {};
template <typename IntT1, typename IntT2, IntT1... Ints1, IntT2... Ints2>
struct concat_int_sequence<std::integer_sequence<IntT1, Ints1...>,
                           std::integer_sequence<IntT2, Ints2...>> {
  using IntT = decltype(std::declval<IntT1>() + std::declval<IntT2>());
  using type = std::integer_sequence<IntT, (IntT)Ints1..., (IntT)Ints2...>;
};
template <typename T, typename U>
using concat_int_sequence_t = typename concat_int_sequence<T, U>::type;

template <typename T, typename U>
struct concat_string_literal_sequence {};
template <StringLiteral... S1, StringLiteral... S2>
struct concat_string_literal_sequence<StringLiteralSequence<S1...>, StringLiteralSequence<S2...>> {
  using type = StringLiteralSequence<S1..., S2...>;
};
template <typename T, typename U>
using concat_string_literal_sequence_t = typename concat_string_literal_sequence<T, U>::type;

template <typename T, typename U>
struct no_overlap {
  static constexpr bool value = true;
};
template <typename T>
struct no_overlap<T, StringLiteralSequence<>> {
  static constexpr bool value = true;
};
template <typename T, StringLiteral S, StringLiteral... Ss>
struct no_overlap<T, StringLiteralSequence<S, Ss...>> {
  static constexpr bool value = !string_literal_sequence_contains_v<T, S> &&
                                no_overlap<T, StringLiteralSequence<Ss...>>::value;
};
template <typename T>
struct no_overlap<T, int_sequence<>> {
  static constexpr bool value = true;
};
template <typename T, int I, int... Is>
struct no_overlap<T, int_sequence<I, Is...>> {
  static constexpr bool value =
      !int_sequence_contains_v<T, I> && no_overlap<T, int_sequence<Is...>>::value;
};
template <typename T, typename U>
constexpr bool no_overlap_v = no_overlap<T, U>::value;

/*
 * The following are equivalent:
 *
 * using S = util::int_sequence<1, 23>;
 * auto arr = util::std_array_v<int64_t, S>;
 *
 * and:
 *
 * using S = util::int_sequence<1, 23>;
 * auto arr = std::array<int64_t, 2>{1, 23};
 */
template <typename T, typename S>
struct std_array {};
template <typename T, typename I, I... Ints>
struct std_array<T, std::integer_sequence<I, Ints...>> {
  static constexpr auto value = std::array<T, sizeof...(Ints)>{Ints...};
};
template <typename T, typename S>
static constexpr auto std_array_v = std_array<T, S>::value;

template <typename DerivedPtr, typename Base>
concept is_pointer_derived_from =
    std::is_pointer_v<DerivedPtr> && std::derived_from<Base, std::remove_pointer_t<DerivedPtr>>;

template <typename T, size_t N>
constexpr size_t array_size(const std::array<T, N>&) {
  return N;
}

/*
 * to_std_array<T>() concatenates all its arguments together into a std::array<T, _> and returns it.
 * Its arguments can either be integral types or std::array types.
 *
 * All of the following are equivalent:
 *
 * std::array{1, 2, 3}
 * to_std_array<int>(1, 2, 3UL)
 * to_std_array<int>(std::array{1, 2}, 3)
 * to_std_array<int>(1, std::array{2, 3})
 * to_std_array<int>(1, std::array{2}, std::array{3})
 *
 * The fact that this function is constexpr allows for elegant compile-time constructions of
 * std::array's.
 */
template <typename A, typename... Ts>
constexpr auto to_std_array(const Ts&... ts);

template <typename T, size_t N>
std::string std_array_to_string(const std::array<T, N>& arr, const std::string& left,
                                const std::string& delim, const std::string& right);

/*
 * std::array<int, 3> a{1, 2, 3};
 * std::array<int64_t, 3> b = util::array_cast<int64_t>(a);
 */
template <typename T, typename U, size_t N>
std::array<T, N> array_cast(const std::array<U, N>&);

template <typename... T>
size_t tuple_hash(const std::tuple<T...>& arg);

template <size_t size>
uint64_t hash_memory(const void* ptr);

}  // namespace util

#include <inline/util/CppUtil.inl>
