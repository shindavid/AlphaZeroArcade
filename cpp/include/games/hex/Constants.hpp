#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "games/hex/Types.hpp"
#include "util/CppUtil.hpp"

namespace hex {

/*
 * Bit order encoding for the board:
 *
 *                     110 111 112 113 114 115 116 117 118 119 120
 *                    99 100 101 102 103 104 105 106 107 108 109
 *                  88  89  90  91  92  93  94  95  96  97  98
 *                77  78  79  80  81  82  83  84  85  86  87
 *              66  67  68  69  70  71  72  73  74  75  76
 *            55  56  57  58  59  60  61  62  63  64  65
 *          44  45  46  47  48  49  50  51  52  53  54
 *        33  34  35  36  37  38  39  40  41  42  43
 *      22  23  24  25  26  27  28  29  30  31  32
 *    11  12  13  14  15  16  17  18  19  20  21
 *   0   1   2   3   4   5   6   7   8   9  10
 *
 * For human-readable notation purposes:
 *
 *                     A11 B11 C11 D11 E11 F11 G11 H11 I11 J11 K11
 *                   A10 B10 C10 D10 E10 F10 G10 H10 I10 J10 K10
 *                  A9  B9  C9  D9  E9  F9  G9  H9  I9  J9  K9
 *                A8  B8  C8  D8  E8  F8  G8  H8  I8  J8  K8
 *              A7  B7  C7  D7  E7  F7  G7  H7  I7  J7  K7
 *            A6  B6  C6  D6  E6  F6  G6  H6  I6  J6  K6
 *          A5  B5  C5  D5  E5  F5  G5  H5  I5  J5  K5
 *        A4  B4  C4  D4  E4  F4  G4  H4  I4  J4  K4
 *      A3  B3  C3  D3  E3  F3  G3  H3  I3  J3  K3
 *    A2  B2  C2  D2  E2  F2  G2  H2  I2  J2  K2
 *  A1  B1  C1  D1  E1  F1  G1  H1  I1  J1  K1
 *
 * First player (red) aims to connect north to south.
 * Second player (blue) aims to connect west to east.
 */
constexpr int kA1 = 0;
constexpr int kB1 = 1;
constexpr int kC1 = 2;
constexpr int kD1 = 3;
constexpr int kE1 = 4;
constexpr int kF1 = 5;
constexpr int kG1 = 6;
constexpr int kH1 = 7;
constexpr int kI1 = 8;
constexpr int kJ1 = 9;
constexpr int kK1 = 10;
constexpr int kA2 = 11;
constexpr int kB2 = 12;
constexpr int kC2 = 13;
constexpr int kD2 = 14;
constexpr int kE2 = 15;
constexpr int kF2 = 16;
constexpr int kG2 = 17;
constexpr int kH2 = 18;
constexpr int kI2 = 19;
constexpr int kJ2 = 20;
constexpr int kK2 = 21;
constexpr int kA3 = 22;
constexpr int kB3 = 23;
constexpr int kC3 = 24;
constexpr int kD3 = 25;
constexpr int kE3 = 26;
constexpr int kF3 = 27;
constexpr int kG3 = 28;
constexpr int kH3 = 29;
constexpr int kI3 = 30;
constexpr int kJ3 = 31;
constexpr int kK3 = 32;
constexpr int kA4 = 33;
constexpr int kB4 = 34;
constexpr int kC4 = 35;
constexpr int kD4 = 36;
constexpr int kE4 = 37;
constexpr int kF4 = 38;
constexpr int kG4 = 39;
constexpr int kH4 = 40;
constexpr int kI4 = 41;
constexpr int kJ4 = 42;
constexpr int kK4 = 43;
constexpr int kA5 = 44;
constexpr int kB5 = 45;
constexpr int kC5 = 46;
constexpr int kD5 = 47;
constexpr int kE5 = 48;
constexpr int kF5 = 49;
constexpr int kG5 = 50;
constexpr int kH5 = 51;
constexpr int kI5 = 52;
constexpr int kJ5 = 53;
constexpr int kK5 = 54;
constexpr int kA6 = 55;
constexpr int kB6 = 56;
constexpr int kC6 = 57;
constexpr int kD6 = 58;
constexpr int kE6 = 59;
constexpr int kF6 = 60;
constexpr int kG6 = 61;
constexpr int kH6 = 62;
constexpr int kI6 = 63;
constexpr int kJ6 = 64;
constexpr int kK6 = 65;
constexpr int kA7 = 66;
constexpr int kB7 = 67;
constexpr int kC7 = 68;
constexpr int kD7 = 69;
constexpr int kE7 = 70;
constexpr int kF7 = 71;
constexpr int kG7 = 72;
constexpr int kH7 = 73;
constexpr int kI7 = 74;
constexpr int kJ7 = 75;
constexpr int kK7 = 76;
constexpr int kA8 = 77;
constexpr int kB8 = 78;
constexpr int kC8 = 79;
constexpr int kD8 = 80;
constexpr int kE8 = 81;
constexpr int kF8 = 82;
constexpr int kG8 = 83;
constexpr int kH8 = 84;
constexpr int kI8 = 85;
constexpr int kJ8 = 86;
constexpr int kK8 = 87;
constexpr int kA9 = 88;
constexpr int kB9 = 89;
constexpr int kC9 = 90;
constexpr int kD9 = 91;
constexpr int kE9 = 92;
constexpr int kF9 = 93;
constexpr int kG9 = 94;
constexpr int kH9 = 95;
constexpr int kI9 = 96;
constexpr int kJ9 = 97;
constexpr int kK9 = 98;
constexpr int kA10 = 99;
constexpr int kB10 = 100;
constexpr int kC10 = 101;
constexpr int kD10 = 102;
constexpr int kE10 = 103;
constexpr int kF10 = 104;
constexpr int kG10 = 105;
constexpr int kH10 = 106;
constexpr int kI10 = 107;
constexpr int kJ10 = 108;
constexpr int kK10 = 109;
constexpr int kA11 = 110;
constexpr int kB11 = 111;
constexpr int kC11 = 112;
constexpr int kD11 = 113;
constexpr int kE11 = 114;
constexpr int kF11 = 115;
constexpr int kG11 = 116;
constexpr int kH11 = 117;
constexpr int kI11 = 118;
constexpr int kJ11 = 119;
constexpr int kK11 = 120;
constexpr int kSwap = 121;

struct Constants : public core::ConstantsBase {
  static constexpr const char* kGameName = "hex";
  static constexpr int kBoardDim = 11;
  static constexpr int kNumSquares = kBoardDim * kBoardDim;
  static constexpr int KNumActions = kNumSquares + 1;  // +1 for swap action

  using kNumActionsPerMode = util::int_sequence<KNumActions>;
  static constexpr int kNumPlayers = 2;
  static constexpr int kMaxBranchingFactor = KNumActions;

  static constexpr core::seat_index_t kRed = 0;   // connects N to S
  static constexpr core::seat_index_t kBlue = 1;  // connects W to E

  static constexpr core::seat_index_t kFirstPlayer = kRed;
  static constexpr core::seat_index_t kSecondPlayer = kBlue;

  static constexpr char kSeatChars[kNumPlayers] = {'R', 'B'};

  static_assert(sizeof(mask_t) * 8 >= kBoardDim, "mask_t must be large enough to hold a row");
  static_assert(kSwap == kNumSquares, "kSwap must be equal to kNumSquares");
};

}  // namespace hex
