#pragma once

#include "core/BasicTypes.hpp"

#include <cstdint>

namespace a0achess {

const int kNumPlayers = 2;
const int kBoardDim = 8;

const core::seat_index_t kWhite = 0;
const core::seat_index_t kBlack = 1;

const int kNumActions = 1858;         // From lc0
const int kMaxBranchingFactor = 500;  // ChatGPT estimates 250, doubling to be generous

const int kNumPastStatesToEncode = 7;
constexpr int kNumRecentHashesToStore = 8;

// Same as Disservin's chess::Square, but packed into a single byte.
enum class Square : std::uint8_t {
  kA1,
  kB1,
  kC1,
  kD1,
  kE1,
  kF1,
  kG1,
  kH1,
  kA2,
  kB2,
  kC2,
  kD2,
  kE2,
  kF2,
  kG2,
  kH2,
  kA3,
  kB3,
  kC3,
  kD3,
  kE3,
  kF3,
  kG3,
  kH3,
  kA4,
  kB4,
  kC4,
  kD4,
  kE4,
  kF4,
  kG4,
  kH4,
  kA5,
  kB5,
  kC5,
  kD5,
  kE5,
  kF5,
  kG5,
  kH5,
  kA6,
  kB6,
  kC6,
  kD6,
  kE6,
  kF6,
  kG6,
  kH6,
  kA7,
  kB7,
  kC7,
  kD7,
  kE7,
  kF7,
  kG7,
  kH7,
  kA8,
  kB8,
  kC8,
  kD8,
  kE8,
  kF8,
  kG8,
  kH8,
  kNumSquares,
};

inline constexpr uint64_t operator<<(uint64_t lhs, Square rhs) {
  return lhs << static_cast<uint64_t>(rhs);
}

enum class CastlingRightBit : std::uint8_t {
  kWhiteKingSide = 0,
  kWhiteQueenSide = 1,
  kBlackKingSide = 2,
  kBlackQueenSide = 3,
};

inline constexpr uint8_t operator<<(uint8_t lhs, CastlingRightBit rhs) {
  return lhs << static_cast<uint8_t>(rhs);
}

using CastlingRights = std::uint8_t;  // 4 bits, one for each of the 4 castling rights

const uint64_t kPawnsMask = 0x00FFFFFFFFFFFF00;

}  // namespace a0achess
