#pragma once

#include <cstdint>

namespace a0achess {

using board_mask_t = uint64_t;
using zobrist_hash_t = uint64_t;

// Same as Disservin's chess::Square, but packed into a single byte.
// clang-format off
enum class Square : std::uint8_t {
  kA1, kB1, kC1, kD1, kE1, kF1, kG1, kH1,
  kA2, kB2, kC2, kD2, kE2, kF2, kG2, kH2,
  kA3, kB3, kC3, kD3, kE3, kF3, kG3, kH3,
  kA4, kB4, kC4, kD4, kE4, kF4, kG4, kH4,
  kA5, kB5, kC5, kD5, kE5, kF5, kG5, kH5,
  kA6, kB6, kC6, kD6, kE6, kF6, kG6, kH6,
  kA7, kB7, kC7, kD7, kE7, kF7, kG7, kH7,
  kA8, kB8, kC8, kD8, kE8, kF8, kG8, kH8,
  kNumSquares,
};
// clang-format on

inline constexpr board_mask_t operator<<(board_mask_t lhs, Square rhs) {
  return lhs << static_cast<board_mask_t>(rhs);
}

enum class CastlingRightBit : std::uint8_t {
  kWhiteKingSide = 0,
  kWhiteQueenSide = 1,
  kBlackKingSide = 2,
  kBlackQueenSide = 3,
};

using CastlingRights = std::uint8_t;  // 4 bits, one for each of the 4 castling rights

inline constexpr CastlingRights operator<<(CastlingRights lhs, CastlingRightBit rhs) {
  return lhs << static_cast<CastlingRights>(rhs);
}

}  // namespace a0achess
