#pragma once

#include "games/othello/BasicTypes.hpp"

#include <array>

namespace othello {

namespace detail {

constexpr int idx(int f, int r) { return r * 8 + f; }
constexpr bool in_bounds(int f, int r) { return (unsigned)f < 8 && (unsigned)r < 8; }
constexpr mask_t bit(int f, int r) { return mask_t{1} << idx(f, r); }
constexpr mask_t bit64(int x) { return mask_t{1} << x; }
constexpr line_mask_t bit8(int x) { return line_mask_t{1} << x; }

} // namespace detail

// -------- Single-line builders --------
constexpr mask_t rank_mask(int r) {
  mask_t m = 0;
  for (int f = 0; f < 8; ++f) m |= detail::bit(f, r);
  return m;
}

constexpr mask_t file_mask(int f) {
  mask_t m = 0;
  for (int r = 0; r < 8; ++r) m |= detail::bit(f, r);
  return m;
}

// from A1 to H8
constexpr mask_t SE_diag_mask(int f0, int r0) {
  mask_t m = 0;
  for (int f = f0, r = r0; detail::in_bounds(f, r); ++f, ++r) m |= detail::bit(f, r);
  return m;
}

// from H1 to A8
constexpr mask_t SW_diag_mask(int f0, int r0) {
  mask_t m = 0;
  for (int f = f0, r = r0; detail::in_bounds(f, r); --f, ++r) m |= detail::bit(f, r);
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

constexpr void move_and_flip(int x, line_mask_t& curr_player, line_mask_t& opponent) {
  if (x > 1) {
    int y = x - 1;
    while (y > 0 && (opponent & detail::bit8(y))) --y;
    if (curr_player & detail::bit8(y))
      for (y = x - 1; y > 0 && (opponent & detail::bit8(y)); --y) {
        opponent ^= detail::bit8(y);
        curr_player ^= detail::bit8(y);
      }
  }
  if (x < 6) {
    int y = x + 1;
    while (y < 8 && (opponent & detail::bit8(y))) ++y;
    if (curr_player & detail::bit8(y))
      for (y = x + 1; y < 8 && (opponent & detail::bit8(y)); ++y) {
        opponent ^= detail::bit8(y);
        curr_player ^= detail::bit8(y);
      }
  }
}

constexpr line_mask_t _find_stable_edge(line_mask_t curr_player_mask, line_mask_t opponent_mask,
                                       line_mask_t stable) {
  line_mask_t empty_spaces = ~(curr_player_mask | opponent_mask);
  stable = stable & curr_player_mask;
  if (!stable || !empty_spaces) return stable;

  for (int x = 0; (x < 8); ++x) {
    if (!(empty_spaces & detail::bit8(x))) continue;
    // player plays
    line_mask_t player_mask_after = curr_player_mask | detail::bit8(x);
    line_mask_t opponent_mask_after = opponent_mask;
    move_and_flip(x, player_mask_after, opponent_mask_after);
    stable = _find_stable_edge(player_mask_after, opponent_mask_after, stable);
    if (!stable) return 0;

    // opponent plays
    line_mask_t player_mask_after2 = curr_player_mask;
    line_mask_t opponent_mask_after2 = opponent_mask | detail::bit8(x);
    move_and_flip(x, opponent_mask_after2, player_mask_after2);
    stable = _find_stable_edge(player_mask_after2, opponent_mask_after2, stable);
    if (!stable) return 0;
  }
  return stable;
}

constexpr line_mask_t find_stable_edge(line_mask_t curr_player_mask, line_mask_t opponent_mask) {
  /*
   * Recursively determines the subset of `stable` discs that remain stable
   * along a single edge (rank or file).
   *
   * - `curr_player_mask` : current player’s discs on this edge
   * - `opponent_mask`    : opponent’s discs on this edge
   *
   * At each recursion step, the function explores all possible moves
   * (both current player and opponent), applies flips, and propagates
   * stability forward. Only discs that remain stable under *all*
   * sequences of moves are returned.
   *
   * Returns: mask of stable discs
   */
  return _find_stable_edge(curr_player_mask, opponent_mask, curr_player_mask) |
         _find_stable_edge(opponent_mask, curr_player_mask, opponent_mask);
}

constexpr void to_binary_masks(int ternary, line_mask_t& curr_player_mask,
                               line_mask_t& opponent_mask) {
  curr_player_mask = 0;
  opponent_mask = 0;
  for (int i = 0; i < 8; ++i) {
    int d = ternary % 3;
    ternary /= 3;
    if (d == 1)
      curr_player_mask |= detail::bit8(i);
    else if (d == 2)
      opponent_mask |= detail::bit8(i);
  }
}

constexpr int kMax = 2 * std::pow(3, 7);  // 2 * 3^7 = 4374

constexpr std::array<uint8_t, kMax> build_stability_array() {
  std::array<uint8_t, kMax> a{};
  line_mask_t curr_player;
  line_mask_t opponent;
  for (int t = 0; t < kMax; ++t) {
    to_binary_masks(t, curr_player, opponent);
    a[t] = find_stable_edge(curr_player, opponent);
  }
  return a;
}

} // namespace othello
