#pragma once

#include <core/BasicTypes.hpp>

namespace c4 {

using column_t = int8_t;
using row_t = int8_t;
using mask_t = uint64_t;
const int kNumColumns = 7;
const int kNumRows = 6;
const int kNumCells = kNumColumns * kNumRows;
const int kMaxMovesPerGame = kNumColumns * kNumRows;
const int kNumPlayers = 2;

const core::seat_index_t kRed = 0;
const core::seat_index_t kYellow = 1;

}  // namespace c4
