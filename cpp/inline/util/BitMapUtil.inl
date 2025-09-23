#include "util/BitMapUtil.hpp"

#include <array>
#include <bit>  // std::byteswap (C++23)
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace bitmap_util {

namespace detail {

// detail::apply_to_all([](uint64_t& m){ f(m); }, mask...);
//
// is equivalent to:
//
// (f(mask), ...);
//
// but with better vectorization potential, by virtue of packing the masks into an array.
template <class Func, class... UInt64T>
  requires((std::is_same_v<std::remove_reference_t<UInt64T>, uint64_t> && ...))
inline void apply_to_all(Func&& f, UInt64T&... masks) {
  constexpr std::size_t N = sizeof...(UInt64T);

  if constexpr (N == 0) {
    return;  // optional: tolerate empty packs
  } else if constexpr (N == 1) {
    // zero-overhead path: no pack/scatter, helps scalar callsites
    auto& only = std::get<0>(std::tie(masks...));
    std::forward<Func>(f)(only);
  } else {
    // 1) Pack
    std::array<uint64_t, N> tmp{masks...};
    // 2) Vectorization-friendly loop
    for (std::size_t i = 0; i < N; ++i) {
      std::forward<Func>(f)(tmp[i]);
    }
    // 3) Scatter
    auto refs = std::tie(masks...);
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((std::get<I>(refs) = tmp[I]), ...);
    }(std::make_index_sequence<N>{});
  }
}

inline void flip_vertical(uint64_t& mask) { mask = std::byteswap(mask); }

inline void mirror_horizontal(uint64_t& mask) {
  constexpr uint64_t k1 = 0x5555555555555555ULL;
  constexpr uint64_t k2 = 0x3333333333333333ULL;
  constexpr uint64_t k4 = 0x0f0f0f0f0f0f0f0fULL;

  uint64_t& x = mask;
  x = ((x >> 1) & k1) | ((x & k1) << 1);
  x = ((x >> 2) & k2) | ((x & k2) << 2);
  x = ((x >> 4) & k4) | ((x & k4) << 4);
}

inline void flip_main_diag(uint64_t& mask) {
  uint64_t t;
  constexpr uint64_t k1 = 0x5500550055005500ULL;
  constexpr uint64_t k2 = 0x3333000033330000ULL;
  constexpr uint64_t k4 = 0x0f0f0f0f00000000ULL;

  uint64_t& x = mask;
  t = k4 & (x ^ (x << 28));
  x ^= t ^ (t >> 28);
  t = k2 & (x ^ (x << 14));
  x ^= t ^ (t >> 14);
  t = k1 & (x ^ (x << 7));
  x ^= t ^ (t >> 7);
}

inline void flip_anti_diag(uint64_t& mask) {
  uint64_t t;
  constexpr uint64_t k1 = 0xaa00aa00aa00aa00ULL;
  constexpr uint64_t k2 = 0xcccc0000cccc0000ULL;
  constexpr uint64_t k4 = 0xf0f0f0f00f0f0f0fULL;

  uint64_t& x = mask;
  t = x ^ (x << 36);
  x ^= k4 & (t ^ (x >> 36));
  t = k2 & (x ^ (x << 18));
  x ^= t ^ (t >> 18);
  t = k1 & (x ^ (x << 9));
  x ^= t ^ (t >> 9);
}

inline void rot90_clockwise(uint64_t& mask) {
  flip_vertical(mask);
  flip_main_diag(mask);
}

inline void rot180(uint64_t& mask) {
  flip_vertical(mask);
  mirror_horizontal(mask);
}

inline void rot270_clockwise(uint64_t& mask) {
  flip_vertical(mask);
  flip_anti_diag(mask);
}

}  // namespace detail

template <class... UInt64T>
  requires((std::is_same_v<std::remove_reference_t<UInt64T>, uint64_t> && ...))
inline void flip_vertical(UInt64T&... mask) {
  detail::apply_to_all([](uint64_t& m) { detail::flip_vertical(m); }, mask...);
}

template <class... UInt64T>
  requires((std::is_same_v<std::remove_reference_t<UInt64T>, uint64_t> && ...))
inline void mirror_horizontal(UInt64T&... mask) {
  detail::apply_to_all([](uint64_t& m) { detail::mirror_horizontal(m); }, mask...);
}

template <class... UInt64T>
  requires((std::is_same_v<std::remove_reference_t<UInt64T>, uint64_t> && ...))
inline void flip_main_diag(UInt64T&... mask) {
  detail::apply_to_all([](uint64_t& m) { detail::flip_main_diag(m); }, mask...);
}

template <class... UInt64T>
  requires((std::is_same_v<std::remove_reference_t<UInt64T>, uint64_t> && ...))
inline void flip_anti_diag(UInt64T&... mask) {
  detail::apply_to_all([](uint64_t& m) { detail::flip_anti_diag(m); }, mask...);
}

template <class... UInt64T>
  requires((std::is_same_v<std::remove_reference_t<UInt64T>, uint64_t> && ...))
inline void rot90_clockwise(UInt64T&... mask) {
  detail::apply_to_all([](uint64_t& m) { detail::rot90_clockwise(m); }, mask...);
}

template <class... UInt64T>
  requires((std::is_same_v<std::remove_reference_t<UInt64T>, uint64_t> && ...))
inline void rot180(UInt64T&... mask) {
  detail::apply_to_all([](uint64_t& m) { detail::rot180(m); }, mask...);
}

template <class... UInt64T>
  requires((std::is_same_v<std::remove_reference_t<UInt64T>, uint64_t> && ...))
inline void rot270_clockwise(UInt64T&... mask) {
  detail::apply_to_all([](uint64_t& m) { detail::rot270_clockwise(m); }, mask...);
}

}  // namespace bitmap_util
