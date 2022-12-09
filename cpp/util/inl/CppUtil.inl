#include <util/CppUtil.hpp>

namespace util {

namespace detail {

/*
 * Adapted from: https://www.linkedin.com/pulse/generic-tuple-hashing-modern-c-alex-dathskovsky/
 */
inline auto lazy_hasher = [](size_t hash, auto&&... values) {
  auto lazy_combiner = [&hash](auto&& val) {
    hash ^= std::hash<std::decay_t<decltype(val)>>{}(val) + 0Xeeffddcc + (hash << 5) + (hash >> 3);
  };
  (lazy_combiner(std::forward<decltype(values)>(values)), ...);
  return hash;
};

struct TupleHasher {
  template<typename... T>
  size_t operator()(const std::tuple<T...>& tup) {
    size_t hash = 0;
    std::apply(lazy_hasher, std::tuple_cat(std::tuple(0), tup));
    return hash;
  }
};

}  // namespace detail

template<typename A>
 constexpr auto to_std_array() {
   return std::array<A, 0>();
 }

 template<typename A, typename T, size_t N>
 constexpr auto to_std_array(const std::array<T, N>& a) {
   std::array<A, N> r;
   for (size_t i=0; i<N; ++i) {
     r[i] = a[i];
   }
   return r;
 }

 template<typename A, typename T>
 constexpr auto to_std_array(T t) {
   std::array<A, 1> r = {A(t)};
   return r;
 }

 template<typename A, typename T, typename... Ts>
 constexpr auto to_std_array(const T& t, const Ts&... ts) {
   auto a = to_std_array<A>(t);
   auto b = to_std_array<A>(ts...);
   std::array<A, array_size(a) + array_size(b)> r;
   std::copy(std::begin(a), std::end(a), std::begin(r));
   std::copy(std::begin(b), std::end(b), std::begin(r)+array_size(a));
   return r;
 }

template<typename T, typename U, size_t N> std::array<T, N> array_cast(const std::array<U, N>& arr) {
  std::array<T, N> out;
  for (size_t i = 0; i < N; ++i) {
    out[i] = arr[i];
  }
  return out;
}

template<typename... T> size_t tuple_hash(const std::tuple<T...>& tup) {
  return detail::TupleHasher{}(tup);
}

}  // namespace util
