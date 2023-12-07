#pragma once

#include <core/BasicTypes.hpp>

namespace tictactoe {

using mask_t = uint16_t;
const int kBoardDimension = 3;
const int kNumCells = kBoardDimension * kBoardDimension;
const int kNumPlayers = 2;

const core::seat_index_t kX = 0;
const core::seat_index_t kO = 1;

}  // namespace tictactoe
