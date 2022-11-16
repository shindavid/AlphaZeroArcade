#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#define CONCAT_HELPER(x, y) x ## y
#define CONCAT(x, y) CONCAT_HELPER(x, y)

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
 * is_std_array_c is for concept requirements for functions that should return std::array's
 */
template<typename T> struct is_std_array { static const bool value = false; };
template<typename T, size_t N> struct is_std_array<std::array<T, N>> { static const bool value = true; };
template<typename T> inline constexpr bool is_std_array_v = is_std_array<T>::value;
template <typename T> concept is_std_array_c = is_std_array_v<T>;

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

}  // namespace util

#include <util/inl/CppUtil.inl>
