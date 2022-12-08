#include <util/CppUtil.hpp>

namespace util {

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


}  // namespace util
