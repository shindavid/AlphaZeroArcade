#pragma once

#include <array>
#include <cstdint>
#include <tuple>
#include <type_traits>

#define CONCAT_HELPER(x, y) x ## y
#define CONCAT(x, y) CONCAT_HELPER(x, y)

#define XSTR(a) STR(a)
#define STR(a) #a

/*
 * Useful macro for constexpr-detection of whether a macro is assigned to 1. This is useful given the behavior of the
 * -D option in py/build.py.
 *
 * #define FOO 1
 * // #define BAR
 *
 * static_assert(IS_MACRO_ASSIGNED_TO_1(FOO))
 * static_assert(!IS_MACRO_ASSIGNED_TO_1(BAR))
 */
#define IS_MACRO_ASSIGNED_TO_1(macro) (XSTR(macro)[0] == '1')

namespace util {

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
 * This function does actually have an implementation, and so you will get a linker error if you try to actually
 * invoke it in other contexts.
 *
 * Adapted from: https://stackoverflow.com/a/69687663/543913
*/
template <class T> std::decay_t<T> decay_copy(T&&);

/*
 * The following are equivalent:
 *
 * using T = util::int_sequence<1, 2, 3>;
 *
 * and:
 *
 * using T = std::integer_sequence<int, 1, 2, 3>;
 */
template<int... Ints> using int_sequence = std::integer_sequence<int, Ints...>;

/*
 * IntSequenceConcept<T> is for concept requirements.
 */
template<typename T> struct is_int_sequence { static const bool value = false; };
template<int... Ints> struct is_int_sequence<int_sequence<Ints...>> { static constexpr bool value = true; };
template<typename T> inline constexpr bool is_int_sequence_v = is_int_sequence<T>::value;
template<typename T> concept IntSequenceConcept = is_int_sequence_v<T>;

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
template<typename T, typename U> struct concat_int_sequence {};
template<int... Ints1, int... Ints2>
struct concat_int_sequence<std::integer_sequence<int, Ints1...>, std::integer_sequence<int, Ints2...>> {
  using type = std::integer_sequence<int, Ints1..., Ints2...>;
};
template<typename T, typename U>
using concat_int_sequence_t = typename concat_int_sequence<T, U>::type;

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
template<typename T, typename S> struct std_array {};
template<typename T, typename I, I... Ints>
struct std_array<T, std::integer_sequence<I, Ints...>> {
  static constexpr auto value = std::array<T, sizeof...(Ints)>{Ints...};
};
template<typename T, typename S>
static constexpr auto std_array_v = std_array<T, S>::value;

template<typename DerivedPtr, typename Base> concept is_pointer_derived_from =
  std::is_pointer_v<DerivedPtr> && std::derived_from<Base, std::remove_pointer_t<DerivedPtr>>;

template<typename T, size_t N>
constexpr size_t array_size(const std::array<T, N>&) { return N; }

/*
 * to_std_array<T>() concatenates all its arguments together into a std::array<T, _> and returns it. Its arguments
 * can either be integral types or std::array types.
 *
 * All of the following are equivalent:
 *
 * std::array{1, 2, 3}
 * to_std_array<int>(1, 2, 3UL)
 * to_std_array<int>(std::array{1, 2}, 3)
 * to_std_array<int>(1, std::array{2, 3})
 * to_std_array<int>(1, std::array{2}, std::array{3})
 *
 * The fact that this function is constexpr allows for elegant compile-time constructions of std::array's.
 */
template<typename A, typename... Ts> constexpr auto to_std_array(const Ts&... ts);

/*
 * std::array<int, 3> a{1, 2, 3};
 * std::array<int64_t, 3> b = util::array_cast<int64_t>(a);
 */
template<typename T, typename U, size_t N> std::array<T, N> array_cast(const std::array<U, N>&);

template<typename... T> size_t tuple_hash(const std::tuple<T...>& arg);

}  // namespace util

#include <util/inl/CppUtil.inl>
