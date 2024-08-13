#pragma once

#include <core/BasicTypes.hpp>

namespace blokus {

using color_t = int8_t;
using piece_index_t = int8_t;
using piece_orientation_index_t = int8_t;

const color_t kBlue = 0;
const color_t kYellow = 1;
const color_t kRed = 2;
const color_t kGreen = 3;

const int kBoardDimension = 20;
const int kNumCells = kBoardDimension * kBoardDimension;

const piece_orientation_index_t kNumPieceOrientations = 91;

using column_mask_t = uint8_t;
using group_subset_t = uint8_t;

const group_subset_t gC1 = 0b00000001;
const group_subset_t gC2 = 0b00000101;
const group_subset_t gC4 = 0b00001111;
const group_subset_t gD2 = 0b00110011;
const group_subset_t gD4 = 0b11111111;

}  // namespace blokus
