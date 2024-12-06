#pragma once

#include <optional>
#include <type_traits>

namespace util {

/*
 * IndexedDispatcher<N> is a class that allows you to dispatch to one of N functions based on a
 * runtime integer index.
 *
 * Example usage:
 *
 * int i = util::IndexedDispatcher<3>::call(n, [&](auto index) {
 *   constexpr int I = decltype(index)::value;  // here I will be equal to n
 *   return I;
 * });
 */
template <int N>
struct IndexedDispatcher {
  // n is guaranteed to be in the range [0, N)
  template <typename F>
  static auto call(int n, F&& f) {
    return call_impl(n, std::forward<F>(f), std::make_integer_sequence<int, N>{});
  }

 private:
  template <typename F, int... I>
  static auto call_impl(int n, F&& f, std::integer_sequence<int, I...>) {
    // Deduce the return type from the first index (0).
    using R = std::invoke_result_t<F, std::integral_constant<int, 0>>;

    // Check that for all I, std::invoke_result_t<F, std::integral_constant<int, I>> == R
    static_assert(
        (std::is_same<R, std::invoke_result_t<F, std::integral_constant<int, I>>>::value && ...),
        "All return types of f must be the same for each index I.");

    // Create a static table of function pointers returning R.
    using call_type = R (*)(F&&);
    static constexpr call_type table[] = {
        [](F&& ff) -> R { return ff(std::integral_constant<int, I>{}); }...};

    return table[n](std::forward<F>(f));
  }
};

// Specialization for N = 1, taking advantage of the assumption that n is assumed to be in the
// range [0, N).
template <>
struct IndexedDispatcher<1> {
  template <typename F>
  static auto call(int n, F&& f) {
    return f(std::integral_constant<int, 0>{});
  }
};

}  // namespace util
