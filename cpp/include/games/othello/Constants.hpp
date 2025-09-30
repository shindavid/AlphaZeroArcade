#pragma once

#include "core/BasicTypes.hpp"

#include <cstdint>

/*
 * Bit order encoding for the board:
 *
 *  0  1  2  3  4  5  6  7
 *  8  9 10 11 12 13 14 15
 * 16 17 18 19 20 21 22 23
 * 24 25 26 27 28 29 30 31
 * 32 33 34 35 36 37 38 39
 * 40 41 42 43 44 45 46 47
 * 48 49 50 51 52 53 54 55
 * 56 57 58 59 60 61 62 63
 *
 * For human-readable notation purposes:
 *
 * A1 B1 C1 D1 E1 F1 G1 H1
 * A2 B2 C2 D2 E2 F2 G2 H2
 * A3 B3 C3 D3 E3 F3 G3 H3
 * A4 B4 C4 D4 E4 F4 G4 H4
 * A5 B5 C5 D5 E5 F5 G5 H5
 * A6 B6 C6 D6 E6 F6 G6 H6
 * A7 B7 C7 D7 E7 F7 G7 H7
 * A8 B8 C8 D8 E8 F8 G8 H8
 *
 * I wanted to have row 1 at the bottom and row 8 at the top like in chess. But every resource I
 * look at on the web appears to go with the above representation.
 */
namespace othello {

using column_t = int8_t;
using row_t = int8_t;
using mask_t = uint64_t;

const int kNumPlayers = 2;
const int kBoardDimension = 8;
const int kNumCells = kBoardDimension * kBoardDimension;
const int kNumStartingPieces = 4;
const mask_t kCompleteBoardMask = ~0ULL;

const int kA1 = 0;
const int kB1 = 1;
const int kC1 = 2;
const int kD1 = 3;
const int kE1 = 4;
const int kF1 = 5;
const int kG1 = 6;
const int kH1 = 7;
const int kA2 = 8;
const int kB2 = 9;
const int kC2 = 10;
const int kD2 = 11;
const int kE2 = 12;
const int kF2 = 13;
const int kG2 = 14;
const int kH2 = 15;
const int kA3 = 16;
const int kB3 = 17;
const int kC3 = 18;
const int kD3 = 19;
const int kE3 = 20;
const int kF3 = 21;
const int kG3 = 22;
const int kH3 = 23;
const int kA4 = 24;
const int kB4 = 25;
const int kC4 = 26;
const int kD4 = 27;
const int kE4 = 28;
const int kF4 = 29;
const int kG4 = 30;
const int kH4 = 31;
const int kA5 = 32;
const int kB5 = 33;
const int kC5 = 34;
const int kD5 = 35;
const int kE5 = 36;
const int kF5 = 37;
const int kG5 = 38;
const int kH5 = 39;
const int kA6 = 40;
const int kB6 = 41;
const int kC6 = 42;
const int kD6 = 43;
const int kE6 = 44;
const int kF6 = 45;
const int kG6 = 46;
const int kH6 = 47;
const int kA7 = 48;
const int kB7 = 49;
const int kC7 = 50;
const int kD7 = 51;
const int kE7 = 52;
const int kF7 = 53;
const int kG7 = 54;
const int kH7 = 55;
const int kA8 = 56;
const int kB8 = 57;
const int kC8 = 58;
const int kD8 = 59;
const int kE8 = 60;
const int kF8 = 61;
const int kG8 = 62;
const int kH8 = 63;
const int kPass = 64;

const int kStartingWhite1 = kD4;
const int kStartingWhite2 = kE5;
const int kStartingBlack1 = kE4;
const int kStartingBlack2 = kD5;

const mask_t kStartingWhiteMask = 1ULL << kStartingWhite1 | 1ULL << kStartingWhite2;
const mask_t kStartingBlackMask = 1ULL << kStartingBlack1 | 1ULL << kStartingBlack2;

constexpr mask_t kA1Mask = 1ULL << kA1;
constexpr mask_t kH1Mask = 1ULL << kH1;
constexpr mask_t kA8Mask = 1ULL << kA8;
constexpr mask_t kH8Mask = 1ULL << kH8;

constexpr mask_t kRank1Mask = 0x00000000000000FFULL;
constexpr mask_t kRank8Mask = 0xFF00000000000000ULL;
constexpr mask_t kFileAMask = 0x0101010101010101ULL;
constexpr mask_t kFileHMask = 0x8080808080808080ULL;

/*
 * +1 for the pass move.
 *
 * Technically, the 4 central squares are not legal in a standard game of othello, but this slightly
 * slack representation simplifies the code.
 */
const int kNumGlobalActions = kNumCells + 1;

/*
 * This can probably be shrunk, maybe down to the 22-34 range. But I haven't found any proof of an
 * upper bound, so being conservative for now.
 *
 * https://puzzling.stackexchange.com/a/102017/18525
 */
const int kMaxNumLocalActions = 40;

const int kTypicalNumMovesPerGame = kNumCells - kNumStartingPieces;

const core::seat_index_t kBlack = 0;
const core::seat_index_t kWhite = 1;
const core::seat_index_t kStartingColor = kBlack;

// A1 = (file=0, rank=0) = bit 0; file→ +1, rank↑ +8
constexpr int idx(int f, int r) { return r * 8 + f; }
constexpr bool in_bounds(int f, int r) { return (unsigned)f < 8 && (unsigned)r < 8; }
constexpr mask_t bit(int f, int r) { return mask_t{1} << idx(f, r); }

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
constexpr mask_t diagSE_from(int f0, int r0) {
  mask_t m = 0;
  for (int f = f0, r = r0; in_bounds(f, r); ++f, ++r) m |= bit(f, r);
  return m;
}

// from H1 to A8
constexpr mask_t diagSW_from(int f0, int r0) {
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
constexpr std::array<mask_t, 15> make_diagSE() {
  std::array<mask_t, 15> a{};
  int k = 0;
  // starts on bottom row A1..H1
  for (int f = 0; f < 8; ++f) a[k++] = diagSE_from(f, 0);
  // starts on left column A2..A8
  for (int r = 1; r < 8; ++r) a[k++] = diagSE_from(0, r);
  return a;
}
constexpr std::array<mask_t, 15> make_diagSW() {
  std::array<mask_t, 15> a{};
  int k = 0;
  // starts on bottom row H1..A1
  for (int f = 7; f >= 0; --f) a[k++] = diagSW_from(f, 0);
  // starts on right column H2..H8
  for (int r = 1; r < 8; ++r) a[k++] = diagSW_from(7, r);
  return a;
}

inline constexpr auto kRanks = make_ranks();
inline constexpr auto kFiles = make_files();
inline constexpr auto kDiagSE = make_diagSE();
inline constexpr auto kDiagSW = make_diagSW();

static_assert(kRanks[0] == kRank1Mask);
static_assert(kRanks[7] == kRank8Mask);
static_assert(kFiles[0] == kFileAMask);
static_assert(kFiles[7] == kFileHMask);

constexpr uint8_t bit8(int x){ return uint8_t(1u<<x); }

constexpr uint8_t find_edge_stable(uint8_t old_P, uint8_t old_O, uint8_t stable) {
  const uint8_t E = uint8_t(~(old_P | old_O));
  stable = uint8_t(stable & old_P);
  if (!stable || !E) return stable;

  for (int x = 0; x < 8; ++x)
    if (E & bit8(x)) {
      // player plays
      {
        uint8_t P = uint8_t(old_P | bit8(x)), O = old_O;
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
        stable = find_edge_stable(P, O, stable);
        if (!stable) return 0;
      }
      // opponent plays
      {
        uint8_t P = old_P, O = uint8_t(old_O | bit8(x));
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
        stable = find_edge_stable(P, O, stable);
        if (!stable) return 0;
      }
    }
  return stable;
}

inline constexpr auto EDGE_STABILITY = [] {
  std::array<uint8_t, 256 * 256> t{};
  for (int P = 0; P < 256; ++P)
    for (int O = 0; O < 256; ++O)
      t[size_t(P) * 256u + size_t(O)] =
        (P & O) ? 0u : find_edge_stable(uint8_t(P), uint8_t(O), uint8_t(P));
  return t;
}();

}  // namespace othello
