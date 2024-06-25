#include <util/BitMapUtil.hpp>

#include <bit>

namespace bitmap_util {

inline void flip_vertical(uint64_t& mask) {
  mask = __builtin_bswap64(mask);
}

inline void mirror_horizontal(uint64_t& mask) {
  constexpr uint64_t k1 = 0x5555555555555555UL;
  constexpr uint64_t k2 = 0x3333333333333333UL;
  constexpr uint64_t k4 = 0x0f0f0f0f0f0f0f0fUL;

  uint64_t& x = mask;
  x = ((x >> 1) & k1) | ((x & k1) << 1);
  x = ((x >> 2) & k2) | ((x & k2) << 2);
  x = ((x >> 4) & k4) | ((x & k4) << 4);
}

inline void flip_main_diag(uint64_t& mask) {
  uint64_t t;
  constexpr uint64_t k1 = 0x5500550055005500UL;
  constexpr uint64_t k2 = 0x3333000033330000UL;
  constexpr uint64_t k4 = 0x0f0f0f0f00000000UL;

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
  constexpr uint64_t k1 = 0xaa00aa00aa00aa00UL;
  constexpr uint64_t k2 = 0xcccc0000cccc0000UL;
  constexpr uint64_t k4 = 0xf0f0f0f00f0f0f0fUL;

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

}  // namespace bitmap_util
