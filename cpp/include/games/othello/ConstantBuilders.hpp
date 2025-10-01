#pragma once

#include "games/othello/BasicTypes.hpp"

#include <array>

namespace othello {

namespace detail {

constexpr int idx(int f, int r) { return r * 8 + f; }
constexpr bool in_bounds(int f, int r) { return (unsigned)f < 8 && (unsigned)r < 8; }
constexpr mask_t bit(int f, int r) { return mask_t{1} << idx(f, r); }
constexpr line_mask_t bit8(int x) { return line_mask_t{1} << x; }

// -------- Single-line builders --------
constexpr mask_t rank_mask(int r) {
  mask_t m = 0;
  for (int f = 0; f < 8; ++f) m |= bit(f, r);
  return m;
}

constexpr mask_t file_mask(int f) {
  mask_t m = 0;
  for (int r = 0; r < 8; ++r) m |= bit(f, r);
  return m;
}

// from A1 to H8
constexpr mask_t SE_diag_mask(int f0, int r0) {
  mask_t m = 0;
  for (int f = f0, r = r0; in_bounds(f, r); ++f, ++r) m |= bit(f, r);
  return m;
}

// from H1 to A8
constexpr mask_t SW_diag_mask(int f0, int r0) {
  mask_t m = 0;
  for (int f = f0, r = r0; in_bounds(f, r); --f, ++r) m |= bit(f, r);
  return m;
}

// -------- Array builders --------
constexpr std::array<mask_t, 8> make_ranks() {
  std::array<mask_t, 8> a{};
  for (int r = 0; r < 8; ++r) a[r] = rank_mask(r);
  return a;
}

constexpr std::array<mask_t, 8> make_files() {
  std::array<mask_t, 8> a{};
  for (int f = 0; f < 8; ++f) a[f] = file_mask(f);
  return a;
}

constexpr std::array<mask_t, 15> make_SE_diags() {
  std::array<mask_t, 15> a{};
  int k = 0;
  // starts on bottom row A1..H1
  for (int f = 0; f < 8; ++f) a[k++] = SE_diag_mask(f, 0);
  // starts on left column A2..A8
  for (int r = 1; r < 8; ++r) a[k++] = SE_diag_mask(0, r);
  return a;
}

constexpr std::array<mask_t, 15> make_SW_diags() {
  std::array<mask_t, 15> a{};
  int k = 0;
  // starts on bottom row H1..A1
  for (int f = 7; f >= 0; --f) a[k++] = SW_diag_mask(f, 0);
  // starts on right column H2..H8
  for (int r = 1; r < 8; ++r) a[k++] = SW_diag_mask(7, r);
  return a;
}



constexpr uint8_t find_stable_edge(uint8_t curr_player_mask, uint8_t opponent_mask, uint8_t stable) {
  const uint8_t E = uint8_t(~(curr_player_mask | opponent_mask));
  stable = uint8_t(stable & curr_player_mask);
  if (!stable || !E) return stable;

  for (int x = 0; x < 8; ++x)
    if (E & bit8(x)) {
      // player plays
      {
        uint8_t P = uint8_t(curr_player_mask | bit8(x)), O = opponent_mask;
        if (x > 1) {
          int y = x - 1;
          while (y > 0 && (O & bit8(y))) --y;
          if (P & bit8(y))
            for (y = x - 1; y > 0 && (O & bit8(y)); --y) {
              O ^= bit8(y);
              P ^= bit8(y);
            }
        }
        if (x < 6) {
          int y = x + 1;
          while (y < 8 && (O & bit8(y))) ++y;
          if (P & bit8(y))
            for (y = x + 1; y < 8 && (O & bit8(y)); ++y) {
              O ^= bit8(y);
              P ^= bit8(y);
            }
        }
        stable = find_stable_edge(P, O, stable);
        if (!stable) return 0;
      }
      // opponent plays
      {
        uint8_t P = curr_player_mask, O = uint8_t(opponent_mask | bit8(x));
        if (x > 1) {
          int y = x - 1;
          while (y > 0 && (P & bit8(y))) --y;
          if (O & bit8(y))
            for (y = x - 1; y > 0 && (P & bit8(y)); --y) {
              O ^= bit8(y);
              P ^= bit8(y);
            }
        }
        if (x < 6) {
          int y = x + 1;
          while (y < 8 && (P & bit8(y))) ++y;
          if (O & bit8(y))
            for (y = x + 1; y < 8 && (P & bit8(y)); ++y) {
              O ^= bit8(y);
              P ^= bit8(y);
            }
        }
        stable = find_stable_edge(P, O, stable);
        if (!stable) return 0;
      }
    }
  return stable;
}

}  // namespace detail

} // namespace othello
