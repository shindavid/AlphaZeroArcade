#pragma once

#include <cstdint>

#include <core/BasicTypes.hpp>

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

}  // namespace othello
