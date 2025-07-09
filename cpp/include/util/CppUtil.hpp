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
#include <vector>

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

template<typename T>
size_t hash(const T& t) { return std::hash<T>{}(t); }

// Drop-in replacement for mit::mutex that does nothing.
struct dummy_mutex {
  void lock() noexcept {}
  void unlock() noexcept {}
  bool try_lock() noexcept { return true; }
};

// Generic hash function for POD types
template <typename T>
struct PODHash {
  static_assert(std::is_standard_layout_v<T>, "PODHash can only be used with POD types");
  static_assert(std::is_trivial_v<T>, "PODHash can only be used with POD types");
  std::size_t operator()(const T& s) const;
};

// Used for concept definitions, when we want to specify the type of a method argument exactly.
template <class T>
struct strict_type_match_t {
  template <class U>
    requires(std::same_as<std::decay_t<T>, std::decay_t<U>>)
  operator U&();
};

template <typename T>
std::string get_typename() {
  return boost::core::demangle(typeid(T).name());
}
template <typename T>
std::string get_typename(const T& t) {
  return get_typename<T>();
}

class TtyMode {
 public:
  static TtyMode* instance();
  bool get_mode() const { return mode_; }
  void set_mode(bool x) { mode_ = x; }

 private:
  TtyMode();

  static TtyMode* instance_;
  bool mode_;
};

/*
 * Returns true if the output is a terminal, false otherwise.
 *
 * This is useful for determining whether to print color codes.
 *
 * https://stackoverflow.com/a/5157076/543913
 */
inline bool tty_mode() { return TtyMode::instance()->get_mode(); }
inline void set_tty_mode(bool x) { TtyMode::instance()->set_mode(x); }

int64_t constexpr inline s_to_ns(int64_t s) { return s * 1000 * 1000 * 1000; }
int64_t constexpr inline us_to_ns(int64_t us) { return us * 1000; }
int64_t constexpr inline ms_to_ns(int64_t ms) { return ms * 1000 * 1000; }

template<typename Rep, typename Period>
int64_t to_ns(const std::chrono::duration<Rep, Period>& duration) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

/*
 * Usage:
 *
 * int64_t ns = util::ns_since_epoch(std::chrono::steady_clock::now());
 */
template <typename TimePoint>
int64_t ns_since_epoch(const TimePoint&);

inline int64_t ns_since_epoch() { return ns_since_epoch(std::chrono::system_clock::now()); }

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
using concat_int_sequence_t = concat_int_sequence<T, U>::type;

template <typename T, typename U>
struct concat_string_literal_sequence {};
template <StringLiteral... S1, StringLiteral... S2>
struct concat_string_literal_sequence<StringLiteralSequence<S1...>, StringLiteralSequence<S2...>> {
  using type = StringLiteralSequence<S1..., S2...>;
};
template <typename T, typename U>
using concat_string_literal_sequence_t = concat_string_literal_sequence<T, U>::type;

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

// Helper function to generate the array of reciprocal values
template <int N, std::size_t... I>
constexpr std::array<float, N> generateReciprocalArray(std::index_sequence<I...>) {
  return {{1.0f / (I + 1)...}};
}

/*
 * ReciprocalTable<N>::values is an array of size N where values[i] = 1.0 / (i + 1).
 */
template <int N>
struct ReciprocalTable {
  static constexpr std::array<float, N> values =
      generateReciprocalArray<N>(std::make_index_sequence<N>{});

  // Accepts i >= 1, returns 1.0 / i, avoiding a division if i <= N.
  static float get(int i) { return i <= N ? values[i - 1] : 1.0f / i; }
};

/*
 * If N<0, just does vec.push_back(u).
 *
 * If N>=0, then does vec.push_back(u), but pops off the front element first if the current size
 * exceeds N.
 *
 * For non-negative N, this simulates push_back() for a circular buffer of size N+1. This is useful
 * in settings where we want circular buffer mechanics, but where we require the container's
 * logical ordering to match the physical ordering.
 */
template <int N, typename T, typename U>
void stuff_back(std::vector<T>& vec, const U& u);

namespace concepts {

template <typename T>
concept IntSequence = is_int_sequence_v<T>;

template <typename T>
concept StdBitSet = is_bit_set_v<T>;

template<typename T>
concept UsableAsHashMapKey = requires(const T& a, const T& b) {
  { std::hash<T>{}(a) } -> std::convertible_to<size_t>;
  { a == b } -> std::convertible_to<bool>;
};

}  // namespace concepts

/*
 * util::get() can be used to get the k-th element of a std::integer_sequence.
 *
 * Example usage showcasing 6 different ways to use get():
 *
 * using T = std::integer_sequence<int, 1, 2, 3>;
 * constexpr K = 1;
 *
 * constexpr int x1 = util::get(1, T{});
 * constexpr int x2 = util::get(T{}, 1);
 * constexpr int x3 = util::get<T>(1);
 * constexpr int x4 = util::get<1>(T{});
 * constexpr int x5 = util::get<1, T>();
 * constexpr int x6 = util::get<T, 1>();
 */
template <typename T, T... Values>
constexpr int get(std::size_t k, std::integer_sequence<T, Values...>) {
  constexpr T arr[] = {Values...};
  return arr[k];
}

template <typename T, T... Values>
constexpr int get(std::integer_sequence<T, Values...>, std::size_t k) {
  constexpr T arr[] = {Values...};
  return arr[k];
}

template <typename T>
constexpr int get(std::size_t k) {
  return get(k, T{});
}

template <int K, typename T>
constexpr int get(T) {
  return get(K, T{});
}

template <int K, typename T>
constexpr int get() {
  return get(K, T{});
}

template <typename T, int K>
constexpr int get() {
  return get(K, T{});
}

}  // namespace util

namespace std {

// hash for std::tuple
template<typename... Ts>
struct hash<std::tuple<Ts...>> {
  size_t operator()(const std::tuple<Ts...>& tup) const {
    return util::tuple_hash(tup);
  }
};

}  // namespace std

#include <inline/util/CppUtil.inl>
