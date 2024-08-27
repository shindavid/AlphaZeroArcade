#pragma once

#include <core/BasicTypes.hpp>

namespace blokus {

using color_t = int8_t;
using piece_index_t = int8_t;
using piece_orientation_index_t = int8_t;
using piece_orientation_corner_index_t = int16_t;
using diagonal_direction_t = int8_t;
using diagonal_direction_mask_t = int8_t;
using column_mask_t = uint8_t;
using group_subset_t = uint8_t;

// which directions are unblocked (at most 2 can ever be unblocked)
enum corner_constraint_t : uint8_t {
  ccNone = 0,
  ccN = 1,
  ccE = 2,
  ccS = 3,
  ccW = 4,
  ccNE = 5,
  ccSE = 6,
  ccSW = 7,
  ccNW = 8
};

enum drawing_t : uint8_t {
  dBlankSpace = 0,
  dBlueSpace = 1,
  dYellowSpace = 2,
  dRedSpace = 3,
  dGreenSpace = 4,
  dCircle = 5,
  dPlus = 6,
  dStar = 7,
  dTimes = 8,
  dNumDrawings = 9
};

inline drawing_t color_to_drawing(color_t c) { return drawing_t(c + 1); }

const piece_index_t pO1 = 0;
const piece_index_t pI2 = 1;
const piece_index_t pI3 = 2;
const piece_index_t pL3 = 3;
const piece_index_t pI4 = 4;
const piece_index_t pO4 = 5;
const piece_index_t pT4 = 6;
const piece_index_t pL4 = 7;
const piece_index_t pS4 = 8;
const piece_index_t pF5 = 9;
const piece_index_t pI5 = 10;
const piece_index_t pL5 = 11;
const piece_index_t pN5 = 12;
const piece_index_t pP5 = 13;
const piece_index_t pT5 = 14;
const piece_index_t pU5 = 15;
const piece_index_t pV5 = 16;
const piece_index_t pW5 = 17;
const piece_index_t pX5 = 18;
const piece_index_t pY5 = 19;
const piece_index_t pZ5 = 20;
const piece_index_t kNumPieces = 21;

const color_t kBlue = 0;
const color_t kYellow = 1;
const color_t kRed = 2;
const color_t kGreen = 3;
const color_t kNumColors = 4;

const diagonal_direction_t dSW = 0;
const diagonal_direction_t dNW = 1;
const diagonal_direction_t dNE = 2;
const diagonal_direction_t dSE = 3;

const int kNumDiagonalDirections = 4;
const int kMaxPieceHeight = 5;
const int kMaxScore = 63;  // a generous guess
const int kBoardDimension = 20;
const int kNumCells = kBoardDimension * kBoardDimension;
const int kNumPieceOrientations = 91;
const int kNumPieceOrientationCorners = 309;
const int kNumPieceOrientationRowMasks = 1102;
const int kCornerConstraintArraySize = 709;
const int kNumSquaresPerColor = 89;
const int kMiniBoardLookupSize = 91;

/*
 * We split the move into two parts:
 *
 * 1. Location (or pass)
 * 2. Piece/orientation/corner to place on that location
 *
 * To allow an action to be interpretable in a context-free manner, we use disjoint regions of
 * the action space for each part of the move.
 *
 * The first part contributes (kNumCells + 1) values (+1 for pass).
 *
 * The second part contributes kNumPieceOrientationCorners values.
 */
const int kNumActions = kNumCells + kNumPieceOrientationCorners + 1;

const core::action_t kPass = kNumCells;

const group_subset_t gC1 = 0b00000001;
const group_subset_t gC2 = 0b00000101;
const group_subset_t gC4 = 0b00001111;
const group_subset_t gD2 = 0b00110011;
const group_subset_t gD4 = 0b11111111;

}  // namespace blokus
